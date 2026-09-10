# NaFlex 当前主线与分支指南

## 代码与资产的边界

feature/caption-regen 是 Caption 生成/预处理及当时实验代码的快照。
caption_tools/output 生成数据未被该基线分支跟踪；Git 分支切换不等于生成或复制 Caption、Codebook、权重或日志。

codex/naflex-reid 由最新 ddp-global-loss-gather 改名，其历史已包含所有 PR 阶段；无需逐个重复合并。
docs/architecture 已合入作为历史资料，旧文档增加适用范围提示。以后新文档优先随主开发线维护。
main、dev、feature/caption-regen 不因本次整理更新，docs/architecture 保留。

## 阶段继承简图

```text
main → dev
       ├─ docs/architecture ──────────────────────┐
       └─ feature/caption-regen                  │
          → attribute-codebook-pr1               │
          → attribute-codebook-pr2               │
          → codebook-weight-schedule             │
          → codebook-delayed-s1-p2                │
          → attribute-relation-pr3               │
          → config-cleanup                       │
          → naflex-ddp-bs256                      │
          → ddp-global-loss-gather               │
          → codex/naflex-reid ← merge docs ───────┘
```

每阶段可能含多个提交。8个中间 codex 分支头使用 archive/2026-09-10/<去掉codex前缀的阶段名> 标签归档。
日常开发入口为 codex/naflex-reid，不需要8个分支同时维护。

## 当前代码阅读入口

1. tools/train.py：默认配置、YAML/CLI、DDP初始化，构建数据/模型/训练器。
2. data/build.py 与 DataManager：源域训练与目标测试、Caption和属性教师目标。
3. data/collate.py：NaFlex patch输入、mask、spatial_shapes及PID/camera/语义目标。
4. modeling/build.py → modeling/backbones/siglip2.py：视觉tokens与native MAP全局特征。
5. modeling/heads/reid_head.py：全局特征、BN和ID分类头。
6. objectives/reid_objective.py：ID、Triplet及按配置启用的Caption、Codebook KL、Relation。
7. engine/trainer.py：AMP、反向、优化器、调度器及周期评估。
8. engine/evaluator.py → utils/metrics.py：image-only embedding、query/gallery距离、mAP/CMC。

池化后的身份特征是B×D，不能与B×N×D的patch tokens混淆。
两卡global loss使用本卡anchor与全局candidate，每卡相似度B×2B；ID分类无需全局候选集合。
Codebook为源域离线资产，KL对齐同一个桶内的文本与图像code分布。
Relation多卡队列/域同步仍需单独验收，目前双卡Codebook启动器关闭Relation。

## 协议与复现

当前包含6组单源迁移与3组Protocol-2配置。
标准Protocol-2不自动包含pseudo-unseen轮换训练；后者是额外方法。
BS64/30ep和BS256/60ep还存在LR、调度、采样等差异，不能只归因于batch。
目标Caption禁用，测试image-only；具体源目标和combineall读取本次配置。

## 查看分支图

```bash
git log --graph --oneline --decorate --all --date-order
git log --graph --oneline --decorate --all --simplify-by-decoration
git branch -avv
git tag --list 'archive/2026-09-10/*'
git worktree list
gitk --all
```

gitk需要本地Git GUI及桌面。查看历史文件用git show <tag>:<path>。
历史脚本可能带旧分支guard，不能假定detached标签下无需调整即可运行。

本次不迁移/删除数据与结果，不修改已有未提交文件，不启动实验。
Campus同步应另行确认活动实验和工作树状态，目录名无需随分支改名。
