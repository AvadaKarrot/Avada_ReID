# NaFlex Caption-C + Attribute Codebook 延迟启用：S1与Protocol-2运行手册

## 1. 本阶段验证什么

固定训练方法：NaFlex图像塔、冻结SigLIP2文本塔、Caption-C权重0.1、Attribute Codebook权重0.05。epoch 1–5只训练Caption-C，epoch 6起启用Codebook。测试阶段始终image-only，目标域不读取Caption。

已完成的M→MS不重复运行。本流水线运行剩余5个单源方向和3个标准Protocol-2方向，seed均为1234、30 epochs、每5 epochs评估与保存。

单源方向：MS→M、MS→C3、C3→MS、C3→M、M→C3。

Protocol-2方向：M+MS+CS→C3、M+CS+C3→MS、MS+CS+C3→M。

这里的Protocol-2是“三源联合训练→第四域测试”，不是“两个源域训练、第三个源域pseudo-unseen”的元学习版本。

## 2. 实际运行文件

唯一入口脚本：

```text
/root/autodl-tmp/Avada_ReID/TransReID/tools/run_naflex_codebook_delayed_s1_p2.sh
```

训练基础配置：

```text
configs/experiments/siglip2_naflex_caption_attribute_codebook_m_to_ms.yml
```

脚本通过命令行覆盖每组的`DATASETS.SOURCES`、`DATASETS.TARGETS`、Codebook资产路径、输出目录和延迟调度。最终完整命令会写入每组训练日志开头的配置打印中。

## 3. Codebook资产

单源资产采用`domain-mode=camera`，保持与已完成Market Codebook相同的构建协议：

```text
/root/autodl-tmp/precomputed/attribute_codebooks/market1501_pr1
/root/autodl-tmp/precomputed/attribute_codebooks/msmt17_pr1
/root/autodl-tmp/precomputed/attribute_codebooks/cuhk03_pr1
```

Protocol-2资产采用`domain-mode=dataset`和`min-domain-coverage=2`，只保留至少被两个源数据集支持的语义节点：

```text
/root/autodl-tmp/precomputed/attribute_codebooks/market1501_msmt17_cuhksysu_pr1
/root/autodl-tmp/precomputed/attribute_codebooks/market1501_cuhksysu_cuhk03_pr1
/root/autodl-tmp/precomputed/attribute_codebooks/msmt17_cuhksysu_cuhk03_pr1
```

每次训练前都会运行`tools/validate_attribute_codebook.py`，核验哈希、Caption文件和目标域不在`source_datasets`中。

## 4. 如何运行

进入项目：

```bash
cd /root/autodl-tmp/Avada_ReID/TransReID
```

只构建/校验Codebook：

```bash
bash tools/run_naflex_codebook_delayed_s1_p2.sh build
```

构建资产并执行两个真实1-epoch smoke（一个S1、一个P2）：

```bash
bash tools/run_naflex_codebook_delayed_s1_p2.sh smoke
```

构建或复核资产后执行8组正式实验：

```bash
bash tools/run_naflex_codebook_delayed_s1_p2.sh train
```

从资产构建、smoke到正式实验完整执行：

```bash
bash tools/run_naflex_codebook_delayed_s1_p2.sh all
```

正式后台运行推荐命令：

```bash
nohup bash tools/run_naflex_codebook_delayed_s1_p2.sh all \
  >/root/autodl-tmp/logs/codebook_delayed_s1_p2.launch.log 2>&1 \
  </dev/null &
```

## 5. 如何查看进度

总监督日志：

```bash
tail -f /root/autodl-tmp/logs/codebook_delayed_s1_p2/supervisor.log
```

查看当前训练：

```bash
pgrep -af 'tools/train.py|build_attribute_codebook.py'
nvidia-smi
```

查看所有状态：

```bash
find /root/autodl-tmp/logs/codebook_delayed_s1_p2 \
  -type f \( -name 'running.json' -o -name 'complete.json' -o -name 'failed.json' \) \
  -print
```

示例：查看MS→M训练日志：

```bash
tail -f /root/autodl-tmp/logs/codebook_delayed_s1_p2/s1/ms_to_m/train.log
```

示例：查看M+MS+CS→C3训练日志：

```bash
tail -f /root/autodl-tmp/logs/codebook_delayed_s1_p2/p2/m_ms_cs_to_c3/train.log
```

## 6. 输出目录

S1：

```text
/root/autodl-tmp/experiments/attribute_codebook_delayed_s1/<direction>/seed_1234
```

Protocol-2：

```text
/root/autodl-tmp/experiments/protocol2_siglip2_naflex_caption_attribute_codebook_delayed_30ep/<direction>/seed_1234
```

每组成功后只保留`model_best.pth.tar`和`train_log.txt`。`checkpoint_latest`、`model_last`和每5 epochs权重会记录到`deleted_non_best_weights.tsv`后删除。失败任务保留现场，不自动清理或重启。

## 7. 断点续训与安全约束

如果输出目录非空且存在`checkpoint_latest.pth.tar`，脚本会自动增加`SOLVER.RESUME_TRAIN=True`和明确的`RESUME_PATH`。如果目录非空但没有完整续训checkpoint，脚本拒绝覆盖。

脚本要求：分支为`codex/codebook-delayed-s1-p2`、工作树干净、GPU可见、没有其他主训练或Codebook构建进程。任何资产或训练失败都会写`failed.json`并停止后续任务。
