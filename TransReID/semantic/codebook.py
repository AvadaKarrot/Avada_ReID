"""Offline construction of bucket-specific SigLIP2 semantic codebooks."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from data.attribute_store import ATTRIBUTE_BUCKETS, AttributeStore


ATTRIBUTE_TEMPLATES = {
    "upper_clothing": "a person whose upper clothing is {}",
    "lower_clothing": "a person whose lower clothing is {}",
    "footwear": "a person wearing {} on their feet",
    "carried_items": "a person carrying {}",
    "accessories": "a person wearing {}",
    "hair": "a person with {}",
    "distinctive_features": (
        "a person with the distinctive feature {}"
    ),
}


def _capacity(unique_phrases: int, max_codes: int) -> int:
    if unique_phrases <= 0:
        return 0
    return min(
        unique_phrases,
        int(max_codes),
        max(8, int(math.ceil(math.sqrt(unique_phrases)))),
    )


def _weighted_spherical_mean(
    embeddings: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if embeddings.ndim != 2 or not embeddings.shape[0]:
        raise ValueError("embeddings must be a non-empty 2-D tensor")
    if weights is None:
        weights = torch.ones(
            embeddings.shape[0], dtype=embeddings.dtype,
            device=embeddings.device,
        )
    weights = weights.to(device=embeddings.device, dtype=embeddings.dtype)
    if weights.ndim != 1 or weights.shape[0] != embeddings.shape[0]:
        raise ValueError("weights and embeddings differ in length")
    if not bool((weights >= 0).all()) or float(weights.sum()) <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    return F.normalize((embeddings * weights[:, None]).sum(0), dim=0)


def robust_spherical_mean(
    embeddings: torch.Tensor,
    base_weights: torch.Tensor | None = None,
    *,
    huber_delta: float = 0.1,
    iterations: int = 3,
) -> torch.Tensor:
    """Huber-downweighted spherical mean for noisy free-text phrases."""

    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if base_weights is None:
        base_weights = torch.ones(
            embeddings.shape[0], dtype=embeddings.dtype
        )
    base_weights = base_weights.to(embeddings)
    center = _weighted_spherical_mean(embeddings, base_weights)
    for _ in range(max(0, int(iterations))):
        residual = (1.0 - embeddings @ center).clamp_min(1e-8)
        robust = torch.where(
            residual <= huber_delta,
            torch.ones_like(residual),
            huber_delta / residual,
        )
        center = _weighted_spherical_mean(
            embeddings, base_weights * robust
        )
    return center


def spherical_kmeans(
    embeddings: torch.Tensor,
    clusters: int,
    *,
    seed: int = 1234,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dependency-free spherical K-Means used only by the offline builder."""

    x = F.normalize(embeddings.float(), dim=1)
    if x.ndim != 2 or not x.shape[0]:
        raise ValueError("embeddings must be a non-empty 2-D tensor")
    clusters = int(clusters)
    if clusters <= 0 or clusters > x.shape[0]:
        raise ValueError("clusters must be in [1, number of embeddings]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    centers = x[torch.randperm(x.shape[0], generator=generator)[:clusters]]
    assignments = torch.full((x.shape[0],), -1, dtype=torch.long)

    for _ in range(int(max_iterations)):
        similarities = x @ centers.t()
        updated_assignments = similarities.argmax(dim=1)
        if torch.equal(updated_assignments, assignments):
            break
        assignments = updated_assignments
        updated_centers = []
        nearest_similarity = similarities.max(dim=1).values
        replacement_order = torch.argsort(nearest_similarity)
        replacement_offset = 0
        for index in range(clusters):
            members = x[assignments == index]
            if not members.shape[0]:
                replacement = replacement_order[replacement_offset]
                replacement_offset += 1
                updated_centers.append(x[replacement])
            else:
                updated_centers.append(F.normalize(members.mean(0), dim=0))
        updated_centers = torch.stack(updated_centers)
        shift = (1.0 - (centers * updated_centers).sum(1)).max()
        centers = updated_centers
        if float(shift) <= tolerance:
            break
    assignments = (x @ centers.t()).argmax(dim=1)
    return assignments, centers


def _compactness(
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    center: torch.Tensor,
) -> float:
    return float((embeddings[indices] @ center).mean())


def _consolidate_clusters(
    embeddings: torch.Tensor,
    assignments: torch.Tensor,
    *,
    merge_similarity: float,
    max_compactness_drop: float,
) -> list[dict]:
    clusters = {}
    next_id = 0
    for label in assignments.unique(sorted=True).tolist():
        indices = torch.where(assignments == label)[0]
        center = _weighted_spherical_mean(embeddings[indices])
        clusters[next_id] = {
            "indices": indices,
            "center": center,
            "compactness": _compactness(embeddings, indices, center),
        }
        next_id += 1

    blocked = set()
    while len(clusters) > 1:
        ids = sorted(clusters)
        best_pair = None
        best_similarity = -float("inf")
        for offset, left in enumerate(ids):
            for right in ids[offset + 1:]:
                pair = frozenset((left, right))
                if pair in blocked:
                    continue
                similarity = float(
                    clusters[left]["center"] @ clusters[right]["center"]
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (left, right)
        if best_pair is None or best_similarity < merge_similarity:
            break

        left, right = best_pair
        left_cluster = clusters[left]
        right_cluster = clusters[right]
        indices = torch.cat(
            (left_cluster["indices"], right_cluster["indices"])
        ).sort().values
        center = _weighted_spherical_mean(embeddings[indices])
        compactness = _compactness(embeddings, indices, center)
        old_compactness = (
            left_cluster["indices"].numel()
            * left_cluster["compactness"]
            + right_cluster["indices"].numel()
            * right_cluster["compactness"]
        ) / indices.numel()
        degradation = old_compactness - compactness
        if degradation <= max_compactness_drop:
            del clusters[left]
            del clusters[right]
            clusters[next_id] = {
                "indices": indices,
                "center": center,
                "compactness": compactness,
                "merge_similarity": best_similarity,
                "compactness_drop": degradation,
            }
            next_id += 1
        else:
            blocked.add(frozenset((left, right)))
    return [clusters[index] for index in sorted(clusters)]


def build_phrase_bank(
    store: AttributeStore,
    text_encoder,
    *,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    domain_mode: str = "dataset",
) -> dict:
    """Encode each unique normalized phrase once with the frozen text tower."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    if hasattr(text_encoder, "to"):
        text_encoder = text_encoder.to(device)
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()
    buckets = {}
    for bucket in ATTRIBUTE_BUCKETS:
        statistics = store.phrase_statistics(
            bucket, domain_mode=domain_mode
        )
        phrases = tuple(statistics)
        encoded = []
        for start in range(0, len(phrases), batch_size):
            values = [
                ATTRIBUTE_TEMPLATES[bucket].format(phrase)
                for phrase in phrases[start:start + batch_size]
            ]
            with torch.no_grad():
                features = text_encoder(values)
            encoded.append(F.normalize(features.float(), dim=1).cpu())
        embeddings = (
            torch.cat(encoded)
            if encoded
            else torch.empty((0, 0), dtype=torch.float32)
        )
        buckets[bucket] = {
            "phrases": phrases,
            "embeddings": embeddings,
            "counts": torch.tensor(
                [statistics[p]["count"] for p in phrases],
                dtype=torch.int64,
            ),
            "mean_quality": torch.tensor(
                [statistics[p]["mean_quality"] for p in phrases],
                dtype=torch.float32,
            ),
            "domains": tuple(statistics[p]["domains"] for p in phrases),
        }
    return {
        "version": 1,
        "domain_mode": domain_mode,
        "buckets": buckets,
    }


def _domain_statistics(
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    domains: Sequence[Mapping[str, int]],
    quality: torch.Tensor,
    expected_domains: Sequence[str],
) -> tuple[torch.Tensor, float, float, float]:
    domain_centers = []
    active_domains = []
    for domain in expected_domains:
        member_indices = []
        member_weights = []
        for index in indices.tolist():
            count = int(domains[index].get(domain, 0))
            if count:
                member_indices.append(index)
                member_weights.append(float(quality[index]))
        if member_indices:
            active_domains.append(domain)
            selected = torch.tensor(member_indices, dtype=torch.long)
            weights = torch.tensor(member_weights, dtype=torch.float32)
            domain_centers.append(
                robust_spherical_mean(embeddings[selected], weights)
            )
    if not domain_centers:
        raise RuntimeError("A semantic cluster has no domain support")
    stacked = torch.stack(domain_centers)
    anchor = robust_spherical_mean(stacked)
    center_consistency = float((stacked @ anchor).mean())
    if len(expected_domains) <= 1:
        coverage = 1.0
        entropy = 1.0
    else:
        coverage = len(active_domains) / len(expected_domains)
        totals = torch.tensor(
            [
                sum(int(domains[index].get(domain, 0)) for index in indices)
                for domain in active_domains
            ],
            dtype=torch.float32,
        )
        probabilities = totals / totals.sum()
        entropy = float(
            -(probabilities * probabilities.log()).sum()
            / math.log(len(expected_domains))
        )
    return anchor, coverage, entropy, center_consistency


def _seed_stability(
    embeddings: torch.Tensor,
    reference: torch.Tensor,
    capacity: int,
    seeds: Sequence[int],
    merge_similarity: float,
    max_compactness_drop: float,
) -> torch.Tensor:
    if not seeds:
        return torch.ones(reference.shape[0])
    matches = []
    for seed in seeds:
        assignments, _ = spherical_kmeans(
            embeddings, capacity, seed=int(seed)
        )
        clusters = _consolidate_clusters(
            embeddings,
            assignments,
            merge_similarity=merge_similarity,
            max_compactness_drop=max_compactness_drop,
        )
        centers = torch.stack([cluster["center"] for cluster in clusters])
        matches.append((reference @ centers.t()).max(dim=1).values)
    return torch.stack(matches).mean(0)


def build_adaptive_codebook(
    phrase_bank: dict,
    *,
    expected_domains: Sequence[str],
    seed: int = 1234,
    stability_seeds: Sequence[int] = (),
    max_codes: int = 64,
    merge_similarity: float = 0.90,
    max_compactness_drop: float = 0.01,
    min_support: int = 1,
    min_compactness: float = 0.0,
    min_domain_coverage: int = 1,
) -> dict:
    """Discover and consolidate semantic codes independently per bucket."""

    expected_domains = tuple(sorted(set(expected_domains)))
    if not expected_domains:
        raise ValueError("expected_domains must not be empty")
    result = {}
    for bucket in ATTRIBUTE_BUCKETS:
        source = phrase_bank["buckets"][bucket]
        embeddings = source["embeddings"].float()
        if not embeddings.shape[0]:
            result[bucket] = {
                "prototypes": embeddings,
                "anchor_confidence": torch.empty(0),
                "assignments": torch.empty(0, dtype=torch.long),
                "statistics": (),
            }
            continue
        embeddings = F.normalize(embeddings, dim=1)
        capacity = _capacity(embeddings.shape[0], max_codes)
        assignments, _ = spherical_kmeans(
            embeddings, capacity, seed=seed
        )
        clusters = _consolidate_clusters(
            embeddings,
            assignments,
            merge_similarity=merge_similarity,
            max_compactness_drop=max_compactness_drop,
        )
        candidate_centers = torch.stack(
            [cluster["center"] for cluster in clusters]
        )
        stability = _seed_stability(
            embeddings,
            candidate_centers,
            capacity,
            [value for value in stability_seeds if value != seed],
            merge_similarity,
            max_compactness_drop,
        )

        accepted = []
        stats = []
        for cluster_index, cluster in enumerate(clusters):
            indices = cluster["indices"]
            support = int(source["counts"][indices].sum())
            active_domains = {
                domain
                for index in indices.tolist()
                for domain, count in source["domains"][index].items()
                if count
            }
            if (
                support < min_support
                or cluster["compactness"] < min_compactness
                or len(active_domains) < min_domain_coverage
            ):
                continue
            anchor, coverage, entropy, consistency = _domain_statistics(
                embeddings,
                indices,
                source["domains"],
                source["mean_quality"],
                expected_domains,
            )
            quality_confidence = float(
                source["mean_quality"][indices].mean()
            )
            confidence = max(0.0, min(1.0, (
                cluster["compactness"]
                * coverage
                * consistency
                * float(stability[cluster_index])
                * quality_confidence
            )))
            accepted.append((indices, anchor, confidence))
            stats.append(
                {
                    "unique_phrases": int(indices.numel()),
                    "support": support,
                    "compactness": float(cluster["compactness"]),
                    "domain_coverage": coverage,
                    "domain_entropy": entropy,
                    "domain_center_consistency": consistency,
                    "seed_stability": float(stability[cluster_index]),
                    "mean_quality": quality_confidence,
                    "confidence": confidence,
                }
            )
        if not accepted:
            raise RuntimeError(
                f"All candidate semantic codes were filtered for {bucket}"
            )

        prototypes = torch.stack([item[1] for item in accepted])
        final_assignments = (embeddings @ prototypes.t()).argmax(dim=1)
        result[bucket] = {
            "prototypes": prototypes,
            "anchor_confidence": torch.tensor(
                [item[2] for item in accepted], dtype=torch.float32
            ),
            "assignments": final_assignments,
            "statistics": tuple(stats),
            "initial_capacity": capacity,
        }
    return {
        "version": 1,
        "expected_domains": expected_domains,
        "buckets": result,
    }


def validate_codebook_artifact(
    codebook: dict,
    phrase_bank: dict | None = None,
) -> dict:
    errors = []
    summary = {}
    if codebook.get("version") != 1:
        errors.append("unsupported codebook version")
    buckets = codebook.get("buckets", {})
    for bucket in ATTRIBUTE_BUCKETS:
        payload = buckets.get(bucket)
        if not isinstance(payload, dict):
            errors.append(f"missing bucket {bucket}")
            continue
        prototypes = payload.get("prototypes")
        confidence = payload.get("anchor_confidence")
        if not torch.is_tensor(prototypes) or prototypes.ndim != 2:
            errors.append(f"{bucket}: prototypes must be 2-D")
            continue
        if not torch.isfinite(prototypes).all():
            errors.append(f"{bucket}: prototypes contain non-finite values")
        if prototypes.shape[0]:
            norm_error = float((prototypes.norm(dim=1) - 1).abs().max())
            if norm_error > 1e-4:
                errors.append(f"{bucket}: prototypes are not normalized")
        if (
            not torch.is_tensor(confidence)
            or confidence.shape != (prototypes.shape[0],)
        ):
            errors.append(f"{bucket}: confidence shape mismatch")
        elif not bool(((confidence >= 0) & (confidence <= 1)).all()):
            errors.append(f"{bucket}: confidence is outside [0,1]")
        if phrase_bank is not None:
            phrase_payload = phrase_bank.get("buckets", {}).get(bucket, {})
            embeddings = phrase_payload.get("embeddings")
            assignments = payload.get("assignments")
            if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
                errors.append(f"{bucket}: invalid phrase embeddings")
            elif embeddings.shape[0] and prototypes.shape[1] != embeddings.shape[1]:
                errors.append(f"{bucket}: embedding dimension mismatch")
            if (
                not torch.is_tensor(assignments)
                or assignments.shape != (embeddings.shape[0],)
            ):
                errors.append(f"{bucket}: assignment shape mismatch")
            elif assignments.numel() and (
                int(assignments.min()) < 0
                or int(assignments.max()) >= prototypes.shape[0]
            ):
                errors.append(f"{bucket}: assignments are out of range")
        summary[bucket] = {
            "codes": int(prototypes.shape[0]),
            "dimension": int(prototypes.shape[1]),
        }
    return {"valid": not errors, "errors": errors, "summary": summary}
