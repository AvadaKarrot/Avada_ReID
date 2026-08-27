"""Queue-assisted code-to-code topology distillation."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from data.attribute_store import ATTRIBUTE_BUCKETS


class AttributeRelationObjective(nn.Module):
    """Match visual code topology to a frozen text-code topology.

    Current-batch image features retain gradients. Historical queue entries
    are detached coverage context only. Relations are built independently
    inside each attribute bucket and domain.
    """

    def __init__(
        self,
        buckets,
        *,
        image_dim,
        relation_temperature=0.1,
        domain_key="dataset",
        queue_size=256,
        max_domains=32,
        min_code_mass=1.0,
        min_effective_samples=2.0,
        min_active_codes=3,
        min_anchor_confidence=0.0,
    ):
        super().__init__()
        if relation_temperature <= 0:
            raise ValueError("relation_temperature must be positive")
        if domain_key not in {"dataset", "camera", "global"}:
            raise ValueError(
                "domain_key must be 'dataset', 'camera', or 'global'"
            )
        if queue_size <= 0 or max_domains <= 0:
            raise ValueError("queue_size and max_domains must be positive")
        if min_code_mass <= 0 or min_effective_samples <= 0:
            raise ValueError("relation coverage thresholds must be positive")
        if min_active_codes < 2:
            raise ValueError("min_active_codes must be at least 2")

        self.relation_temperature = float(relation_temperature)
        self.domain_key = str(domain_key)
        self.queue_size = int(queue_size)
        self.max_domains = int(max_domains)
        self.min_code_mass = float(min_code_mass)
        self.min_effective_samples = float(min_effective_samples)
        self.min_active_codes = int(min_active_codes)
        self.min_anchor_confidence = float(min_anchor_confidence)
        self.bucket_names = tuple(
            bucket for bucket in ATTRIBUTE_BUCKETS if bucket in buckets
        )

        self.register_buffer(
            "feature_queue",
            torch.zeros(max_domains, queue_size, int(image_dim)),
            persistent=False,
        )
        self.register_buffer(
            "quality_queue",
            torch.zeros(max_domains, queue_size),
            persistent=False,
        )
        self.register_buffer(
            "queue_ptr", torch.zeros(max_domains, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "queue_count", torch.zeros(max_domains, dtype=torch.long),
            persistent=False,
        )
        for index, bucket in enumerate(self.bucket_names):
            prototypes = F.normalize(
                buckets[bucket]["prototypes"].float(), dim=1
            )
            if prototypes.shape[1] != int(image_dim):
                raise ValueError(
                    f"{bucket} code dimension {prototypes.shape[1]} does not "
                    f"match image alignment dimension {image_dim}"
                )
            confidence = buckets[bucket].get("anchor_confidence")
            if confidence is None:
                confidence = torch.ones(prototypes.shape[0])
            confidence = confidence.float().clamp_min(0)
            self.register_buffer(
                f"prototype_{index}", prototypes, persistent=False
            )
            self.register_buffer(
                f"confidence_{index}", confidence, persistent=False
            )
            self.register_buffer(
                f"target_queue_{index}",
                torch.zeros(max_domains, queue_size, prototypes.shape[0]),
                persistent=False,
            )
            self.register_buffer(
                f"mask_queue_{index}",
                torch.zeros(max_domains, queue_size, dtype=torch.bool),
                persistent=False,
            )
        self.last_metrics = {}

    def _domain_ids(self, batch, device):
        if self.domain_key == "global":
            return torch.zeros(
                batch["pids"].shape[0], dtype=torch.long, device=device
            )
        key = "dataset_ids" if self.domain_key == "dataset" else "camids"
        values = batch.get(key)
        if values is None:
            raise ValueError(f"Attribute relation requires batch[{key!r}]")
        return torch.as_tensor(values, dtype=torch.long, device=device)

    @staticmethod
    def _write_ring(buffer, domain, pointer, values):
        size = buffer.shape[1]
        count = values.shape[0]
        first = min(count, size - pointer)
        buffer[domain, pointer : pointer + first].copy_(values[:first])
        if first < count:
            buffer[domain, : count - first].copy_(values[first:])

    @torch.no_grad()
    def _enqueue(self, images, attribute_targets, domain_ids):
        quality = attribute_targets["quality"].to(images.device).float()
        for domain_tensor in domain_ids.unique():
            domain = int(domain_tensor.item())
            if domain < 0 or domain >= self.max_domains:
                raise ValueError(
                    f"domain id {domain} exceeds configured [0, "
                    f"{self.max_domains - 1}] range"
                )
            select = domain_ids == domain
            features = images[select].detach()
            domain_quality = quality[select].detach()
            if features.shape[0] > self.queue_size:
                features = features[-self.queue_size :]
                domain_quality = domain_quality[-self.queue_size :]
            pointer = int(self.queue_ptr[domain].item())
            self._write_ring(
                self.feature_queue, domain, pointer, features
            )
            self._write_ring(
                self.quality_queue, domain, pointer, domain_quality
            )
            for index, bucket in enumerate(self.bucket_names):
                target = attribute_targets["distributions"][bucket].to(
                    images.device
                ).float()[select]
                mask = attribute_targets["mask"][bucket].to(
                    images.device
                ).bool()[select]
                if target.shape[0] > self.queue_size:
                    target = target[-self.queue_size :]
                    mask = mask[-self.queue_size :]
                self._write_ring(
                    getattr(self, f"target_queue_{index}"),
                    domain,
                    pointer,
                    target.detach(),
                )
                self._write_ring(
                    getattr(self, f"mask_queue_{index}"),
                    domain,
                    pointer,
                    mask.detach(),
                )
            inserted = features.shape[0]
            self.queue_ptr[domain] = (pointer + inserted) % self.queue_size
            self.queue_count[domain] = min(
                self.queue_size,
                int(self.queue_count[domain].item()) + inserted,
            )

    def forward(self, image_features, attribute_targets, batch):
        if attribute_targets is None:
            return image_features.sum() * 0.0
        images = F.normalize(image_features.float(), dim=1)
        domain_ids = self._domain_ids(batch, images.device)
        quality = attribute_targets["quality"].to(images.device).float()

        loss_sum = images.new_zeros(())
        loss_weight = images.new_zeros(())
        graph_count = 0
        pair_count = 0
        skipped_count = 0
        active_code_sum = 0
        teacher_entropy_sum = images.new_zeros(())
        student_entropy_sum = images.new_zeros(())
        active_domains = set()

        for domain_tensor in domain_ids.unique():
            domain = int(domain_tensor.item())
            if domain < 0 or domain >= self.max_domains:
                raise ValueError(
                    f"domain id {domain} exceeds configured max_domains"
                )
            current_select = domain_ids == domain
            queue_count = int(self.queue_count[domain].item())
            queue_features = self.feature_queue[
                domain, :queue_count
            ].detach()
            queue_quality = self.quality_queue[
                domain, :queue_count
            ].detach()

            for index, bucket in enumerate(self.bucket_names):
                current_mask = attribute_targets["mask"][bucket].to(
                    images.device
                ).bool()[current_select]
                if not current_mask.any():
                    skipped_count += 1
                    continue
                current_target = attribute_targets["distributions"][
                    bucket
                ].to(images.device).float()[current_select]
                queue_target = getattr(
                    self, f"target_queue_{index}"
                )[domain, :queue_count].detach()
                queue_mask = getattr(
                    self, f"mask_queue_{index}"
                )[domain, :queue_count].detach()

                features = torch.cat(
                    [images[current_select], queue_features], dim=0
                )
                targets = torch.cat(
                    [current_target, queue_target], dim=0
                )
                masks = torch.cat([current_mask, queue_mask], dim=0)
                qualities = torch.cat(
                    [quality[current_select], queue_quality], dim=0
                )
                weights = (
                    targets
                    * masks.float().unsqueeze(1)
                    * qualities.unsqueeze(1)
                )
                mass = weights.sum(dim=0)
                effective_samples = mass.square() / (
                    weights.square().sum(dim=0).clamp_min(1e-12)
                )
                confidence = getattr(self, f"confidence_{index}")
                active = (
                    (mass >= self.min_code_mass)
                    & (
                        effective_samples
                        >= self.min_effective_samples
                    )
                    & (confidence >= self.min_anchor_confidence)
                )
                active_count = int(active.sum().item())
                if active_count < self.min_active_codes:
                    skipped_count += 1
                    continue

                visual_codes = F.normalize(
                    weights[:, active].t() @ features
                    / mass[active].unsqueeze(1).clamp_min(1e-12),
                    dim=1,
                )
                text_codes = getattr(self, f"prototype_{index}")[active]
                active_confidence = confidence[active].clamp_min(1e-6)
                teacher_relation = text_codes @ text_codes.t()
                student_relation = visual_codes @ visual_codes.t()
                diagonal = torch.eye(
                    active_count, dtype=torch.bool, device=images.device
                )
                confidence_prior = active_confidence.log().unsqueeze(0)
                teacher_logits = (
                    teacher_relation / self.relation_temperature
                    + confidence_prior
                ).masked_fill(diagonal, -1e4)
                student_logits = (
                    student_relation / self.relation_temperature
                    + confidence_prior
                ).masked_fill(diagonal, -1e4)
                teacher_prob = F.softmax(teacher_logits, dim=1)
                student_log_prob = F.log_softmax(student_logits, dim=1)
                per_row = F.kl_div(
                    student_log_prob, teacher_prob, reduction="none"
                ).sum(dim=1)
                loss_sum = loss_sum + (
                    per_row * active_confidence
                ).sum()
                loss_weight = loss_weight + active_confidence.sum()

                entropy_norm = math.log(max(active_count - 1, 2))
                teacher_entropy_sum = teacher_entropy_sum + (
                    -(teacher_prob.clamp_min(1e-12)
                      * teacher_prob.clamp_min(1e-12).log()).sum(dim=1)
                    / entropy_norm
                ).mean()
                student_prob = student_log_prob.exp()
                student_entropy_sum = student_entropy_sum + (
                    -(student_prob.clamp_min(1e-12)
                      * student_log_prob).sum(dim=1)
                    / entropy_norm
                ).mean()
                graph_count += 1
                active_domains.add(domain)
                active_code_sum += active_count
                pair_count += active_count * (active_count - 1) // 2

        self._enqueue(images, attribute_targets, domain_ids)
        denominator = loss_weight.clamp_min(1e-12)
        loss = loss_sum / denominator
        metric_denominator = max(graph_count, 1)
        self.last_metrics = {
            "relation_graphs": images.new_tensor(float(graph_count)),
            "relation_active_domains": images.new_tensor(
                float(len(active_domains))
            ),
            "relation_skipped_graphs": images.new_tensor(
                float(skipped_count)
            ),
            "relation_active_codes": images.new_tensor(
                float(active_code_sum) / metric_denominator
            ),
            "relation_active_pairs": images.new_tensor(
                float(pair_count) / metric_denominator
            ),
            "relation_teacher_entropy": (
                teacher_entropy_sum / metric_denominator
            ).detach(),
            "relation_student_entropy": (
                student_entropy_sum / metric_denominator
            ).detach(),
            "relation_queue_fill": (
                self.queue_count.float().sum()
                / float(self.max_domains * self.queue_size)
            ).detach(),
        }
        if graph_count == 0:
            return image_features.sum() * 0.0
        return loss
