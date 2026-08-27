# PR3：Relation Coverage 与 Code-to-Code 拓扑蒸馏

## 目标与边界

PR3在现有`Caption-C + Instance Attribute L_code`之上增加训练期的桶内关系蒸馏。冻结SigLIP2文本塔、离线Codebook和目标域image-only测试协议不变；不读取目标域Caption，不更新Anchor，不做跨bucket关系。

总损失候选为：

```text
L = L_ID + L_Tri + 0.1 L_Caption-C + 0.05 L_code + lambda_R L_relation
```

PR3配置先给`lambda_R=0.01`作为smoke与小规模消融起点，不代表正式最优值。

## Phrase Bank与离线Codebook保持不变

当前真实`phrase_bank.pt`按bucket保存规范化后的unique phrase，而不是为每张图重复存同一个embedding；同时保留`counts`、`mean_quality`和各源域频次`domains`。因此重复phrase的支持度信息没有丢失，但“出现次数多”不被直接等同为“语义更重要”。

PR3不修改Anchor构建权重，避免把Relation变量与Codebook重建变量混在同一次实验。它直接复用现有`codebook.pt`的Anchor与confidence，以及每张源图像已预计算的`q^T`。

## 为什么采用Queue-assisted，而不是纯Batch

使用真实`RandomIdentitySampler`各抽取100个batch诊断：

- S1（MS→M）：每批camera数中位12，但大多数camera只有1–10个样本；`upper_clothing`按camera的活跃code通常只有0–2个，无法形成稳定拓扑。
- P2（M+MS+CS→C3）：每批源数据集数均值2.31、中位2、p10为1；三个源域每批`upper_clothing`有效样本中位分别约4、8、55，前两个域的活跃code中位只有1、2。

因此纯batch的domain-conditioned Code图会频繁没有有效pair。PR3使用历史队列补足覆盖，但当前batch特征始终保留梯度；队列内容全部`detach`，只作为统计上下文。

诊断入口：

```text
tools/diagnose_attribute_relation_coverage.py
```

诊断结果：

```text
/root/autodl-tmp/logs/attribute_relation_pr3/coverage_s1_ms_to_m.json
/root/autodl-tmp/logs/attribute_relation_pr3/coverage_p2_m_ms_cs_to_c3.json
```

## 算法

对bucket `b`、domain `d`，离线文本Anchor为`p_k^b`，实例文本对Codebook的固定soft assignment为`q_i^T(k)`。组合当前batch和同域队列后：

```text
w_ik = valid_i^b * quality_i * q_i^T(k)
mass_k = sum_i w_ik
ESS_k = mass_k^2 / sum_i w_ik^2
v_k,d^b = Normalize(sum_i w_ik z_i / mass_k)
```

只有同时满足`mass_k >= 1`、`ESS_k >= 2`和Anchor confidence阈值的Code才进入图；少于3个有效Code时跳过该bucket-domain图。

文本与图像拓扑：

```text
R_T(k,j) = cosine(p_k, p_j)
R_I(k,j) = cosine(v_k,d, v_j,d)
```

去除对角线后，对每个Code的邻居分布做温度softmax，并计算：

```text
L_relation = weighted mean_k KL(softmax(R_T(k,:)/tau_R)
                                || softmax(R_I(k,:)/tau_R))
```

Anchor confidence既作为邻居先验，也作为行loss权重。S1使用`DOMAIN_KEY=camera`；Protocol-2使用`DOMAIN_KEY=dataset`。这是让每个源域/相机中相同共享语义Code的视觉拓扑接近同一文本拓扑，而不是直接比较不对应的PID。

## 文件

```text
objectives/losses/attribute_relation.py
objectives/build.py
objectives/reid_objective.py
config/defaults.py
utils/config_validation.py
tests/test_attribute_relation.py
configs/experiments/siglip2_naflex_caption_attribute_codebook_relation_m_to_ms.yml
tools/diagnose_attribute_relation_coverage.py
```

## 运行

单元测试：

```bash
cd /root/autodl-tmp/Avada_ReID/TransReID
/root/miniconda3/bin/python -m unittest \
  tests.test_attribute_relation tests.test_attribute_codebook_pr2
```

S1真实1-epoch smoke（使relation从epoch 1启用）：

```bash
/root/miniconda3/bin/python tools/train.py \
  --config_file configs/experiments/siglip2_naflex_caption_attribute_codebook_relation_m_to_ms.yml \
  SOLVER.MAX_EPOCHS 1 SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 32 \
  DATALOADER.NUM_WORKERS 2 SOLVER.EVAL_PERIOD 1 SOLVER.CHECKPOINT_PERIOD 1 \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.START_EPOCH 1 \
  OBJECTIVE.ATTRIBUTE_RELATION.START_EPOCH 1 \
  OBJECTIVE.ATTRIBUTE_RELATION.QUEUE_SIZE 64 \
  OUTPUT_DIR /root/autodl-tmp/experiments/attribute_relation_pr3_smoke/s1_m_to_ms
```

Protocol-2 smoke必须覆盖资产和domain key：

```bash
/root/miniconda3/bin/python tools/train.py \
  --config_file configs/experiments/siglip2_naflex_caption_attribute_codebook_relation_m_to_ms.yml \
  DATASETS.SOURCES market1501,msmt17,cuhksysu DATASETS.TARGETS cuhk03 \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.PHRASE_BANK /root/autodl-tmp/precomputed/attribute_codebooks/market1501_msmt17_cuhksysu_pr1/phrase_bank.pt \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.CODEBOOK /root/autodl-tmp/precomputed/attribute_codebooks/market1501_msmt17_cuhksysu_pr1/codebook.pt \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.MANIFEST /root/autodl-tmp/precomputed/attribute_codebooks/market1501_msmt17_cuhksysu_pr1/manifest.json \
  OBJECTIVE.ATTRIBUTE_RELATION.DOMAIN_KEY dataset \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.START_EPOCH 1 OBJECTIVE.ATTRIBUTE_RELATION.START_EPOCH 1 \
  OBJECTIVE.ATTRIBUTE_RELATION.QUEUE_SIZE 64 \
  SOLVER.MAX_EPOCHS 1 SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 32 \
  DATALOADER.NUM_WORKERS 2 SOLVER.EVAL_PERIOD 1 SOLVER.CHECKPOINT_PERIOD 1 \
  OUTPUT_DIR /root/autodl-tmp/experiments/attribute_relation_pr3_smoke/p2_m_ms_cs_to_c3
```

训练日志必须出现`attribute_relation`、`relation_graphs`、`relation_active_domains`、`relation_active_codes`、`relation_active_pairs`、teacher/student entropy和queue fill。正式30-epoch实验需用户再次确认后才启动。
