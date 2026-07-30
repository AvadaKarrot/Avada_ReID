# 05 实验协议：DG-ReID Protocol-1/2/3 标准定义

> 权威来源：*Domain Generalization for Person Re-identification: A Survey Towards Domain-Agnostic Person Matching*（Lee et al., arXiv:2506.12413v1, §5.3 Evaluation Protocols），与 ACL(ECCV'22)、CILP-FGDI(arXiv:2501.16065) 等工作的定义一致。
> 本项目决定：**只做 Protocol-1 和 Protocol-2**。

## 数据集缩写

| 缩写 | 数据集 | 角色定位 |
|---|---|---|
| M | Market-1501 | 大型源域 / P2 Target 轮换 |
| MS | MSMT17 | 大型源域 / P2 Target 轮换 |
| C2 | CUHK02 | 仅 P1 源域 |
| C3 | CUHK03 | 大型源域 / P2 Target 轮换 |
| CS | CUHK-SYSU | **永远只当 Source**（见下） |
| — | PRID2011 / GRID / VIPeR / i-LIDS | 小数据集，仅 P1 测试域 |

## Protocol-1（小源域合并 → 小测试集）

- **Source**：M + C2 + C3 + CS 合并训练；**使用源域的全部图像（train + test 子集）**。
- **Target**：PRID2011、GRID、VIPeR、i-LIDS 四个小数据集（完全未见域）。
- **评测**：query/gallery 随机划分 **10 次取平均**，报 mAP + Rank-1。
- 注意：**MSMT17 不参与 P1**。

## Protocol-2（留一法 leave-one-out）

- 四个大数据集 M / MS / C3 / CS 中**选一个当 Target，其余三个当 Source 训练**。
- **Target 轮换池只有 {M, MS, C3}**——CS 被明确排除，原因：CS 只有单摄像头视角，无法进行跨相机匹配评测（cross-camera matching 是 ReID 评测的基本要求）。
- **Source 只用 train 子集**（与 P3 的区别）。
- 训练域与测试域完全不相交，是域偏移最严重的设定，最能检验泛化能力。

## Protocol-3（本项目暂不做，仅备查）

- 与 P2 相同的留一法 Target 轮换；区别仅在于 **Source 用 train+test 全部图像**。
- 更贴近实际部署（各种来源的标注数据全部可用）。

## 对本项目的约束（执行清单）

1. **MSMT17 必须下载**：P2 中既当 Target（Target=MS 轮）又当 Source（其他轮）。
2. **CUHK02 仅服务 P1**：找不到不阻塞 P2；优先级最低。
3. **M / MS / C3 需双边准备**：train（Source 用）+ query/gallery（P2 Target 轮换用），下载保留完整目录结构。
4. **CS 只需 train 部分**（cropped_images 即可），永远不会被评测。
5. **四个小数据集只需 query/gallery**（P1 测试域）；VIPeR/PRID/GRID/i-LIDS 无官方划分的按协议随机划分 10 次取平均。
6. **caption 生成范围待决策**：P1 严格定义要求源域 train+test 全部图像参与训练，而 caption_tools 当前只生成 train split。是否扩展到 query/gallery 取决于对标 baseline（如 ChatAnything）的实际做法——确定对比对象后核对其原文再定。
7. **旧 config 的一对一跨域设定**（如 cuhk03→msmt17）是 2023 年 single-source 遗迹，与本协议体系无关，实验配置需按上表重建。
