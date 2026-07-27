"""caption 生成服务器环境自检脚本。

在推理服务器（Ubuntu 22.04 / Python 3.12 / PyTorch 2.8.0+cu128 /
RTX PRO 6000 96GB Blackwell sm_120 / vLLM）上运行，逐项检查依赖是否就绪，
每项打印 PASS/FAIL，结尾汇总并以退出码标识整体结果（0=全部通过，1=存在失败项）。

用法：
    python check_env.py
"""

from __future__ import annotations

import importlib
import sys

# 依赖的最低版本要求
_MIN_TRANSFORMERS = (4, 56, 0)
_MIN_TIMM = (1, 0, 20)
# 训练侧 DINOv3 backbone 必须在 timm 模型注册表中可用
_DINOV3_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"


def _parse_version(version_str: str) -> tuple[int, ...]:
    """将 '4.56.0' 之类的版本字符串解析为整数元组，忽略预发布后缀。"""
    parts: list[int] = []
    for tok in version_str.split("."):
        digits = ""
        for ch in tok:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_gte(current: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    """语义化版本比较：current >= minimum（按位对齐补齐 0）。"""
    n = max(len(current), len(minimum))
    cur = current + (0,) * (n - len(current))
    req = minimum + (0,) * (n - len(minimum))
    return cur >= req


def check_torch_cuda() -> bool:
    """检查 torch 可导入且 CUDA 可用。"""
    try:
        import torch
    except ImportError:
        print("[FAIL] torch 无法导入")
        return False
    if not torch.cuda.is_available():
        print(f"[FAIL] torch {torch.__version__} 已安装，但 CUDA 不可用")
        return False
    print(f"[PASS] torch {torch.__version__}，CUDA 可用（{torch.version.cuda}）")
    return True


def check_gpu() -> bool:
    """检查 GPU 名称与显存（预期 RTX PRO 6000 96GB）。"""
    try:
        import torch
    except ImportError:
        print("[FAIL] torch 无法导入，跳过 GPU 信息检查")
        return False
    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用，无法查询 GPU 信息")
        return False
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # 96GB 卡允许少量保留显存误差，阈值取 80GB
    ok = total_gb >= 80.0
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] GPU: {name}，显存 {total_gb:.1f} GB（预期 >= 80 GB）")
    return ok


def check_compute_capability() -> bool:
    """检查算力 >= (12,0)（Blackwell sm_120）。"""
    try:
        import torch
    except ImportError:
        print("[FAIL] torch 无法导入，跳过算力检查")
        return False
    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用，无法查询算力")
        return False
    cap = torch.cuda.get_device_capability(0)
    ok = cap >= (12, 0)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] CUDA 算力 sm_{cap[0]}{cap[1]}（{cap}），要求 >= (12, 0) Blackwell")
    return ok


def check_transformers() -> bool:
    """检查 transformers >= 4.56。"""
    try:
        import transformers
    except ImportError:
        print("[FAIL] transformers 无法导入")
        return False
    ver = _parse_version(transformers.__version__)
    ok = _version_gte(ver, _MIN_TRANSFORMERS)
    tag = "PASS" if ok else "FAIL"
    req = ".".join(str(x) for x in _MIN_TRANSFORMERS)
    print(f"[{tag}] transformers {transformers.__version__}（要求 >= {req}）")
    return ok


def check_timm() -> bool:
    """检查 timm >= 1.0.20 且 DINOv3 backbone 在模型注册表中。"""
    try:
        import timm
    except ImportError:
        print("[FAIL] timm 无法导入")
        return False
    ver = _parse_version(timm.__version__)
    req = ".".join(str(x) for x in _MIN_TIMM)
    if not _version_gte(ver, _MIN_TIMM):
        print(f"[FAIL] timm {timm.__version__}（要求 >= {req}）")
        return False
    dinov3_models = timm.list_models("*dinov3*")
    if _DINOV3_MODEL_NAME not in dinov3_models:
        print(
            f"[FAIL] timm {timm.__version__} 满足版本要求，但模型注册表中缺少 "
            f"'{_DINOV3_MODEL_NAME}'（dinov3 相关模型共 {len(dinov3_models)} 个）"
        )
        return False
    print(
        f"[PASS] timm {timm.__version__}，"
        f"'{_DINOV3_MODEL_NAME}' 已在模型注册表中（dinov3 共 {len(dinov3_models)} 个）"
    )
    return True


def check_vllm() -> bool:
    """检查 vllm 可导入并打印版本。"""
    try:
        vllm = importlib.import_module("vllm")
    except ImportError:
        print("[FAIL] vllm 无法导入（Qwen3-VL-32B-Instruct-FP8 推理依赖 vLLM）")
        return False
    version = getattr(vllm, "__version__", "未知版本")
    print(f"[PASS] vllm {version} 可导入")
    return True


def check_misc() -> bool:
    """检查 PIL 与 PyYAML 可用。"""
    ok = True
    try:
        import PIL
        print(f"[PASS] Pillow {PIL.__version__} 可用")
    except ImportError:
        print("[FAIL] PIL（Pillow）无法导入")
        ok = False
    try:
        import yaml
        print(f"[PASS] PyYAML {yaml.__version__} 可用")
    except ImportError:
        print("[FAIL] PyYAML 无法导入")
        ok = False
    return ok


def main() -> int:
    print("=" * 64)
    print("caption 生成服务器环境自检")
    print(f"Python: {sys.version.split()[0]}（{sys.executable}）")
    print("=" * 64)

    checks = [
        ("torch + CUDA", check_torch_cuda),
        ("GPU 名称/显存", check_gpu),
        ("CUDA 算力 (Blackwell)", check_compute_capability),
        ("transformers 版本", check_transformers),
        ("timm 版本 + DINOv3", check_timm),
        ("vllm", check_vllm),
        ("PIL / PyYAML", check_misc),
    ]

    results: list[tuple[str, bool]] = []
    for name, fn in checks:
        try:
            ok = fn()
        except Exception as e:  # 单个检查异常不应中断后续检查
            print(f"[FAIL] {name} 检查过程抛出异常：{e}")
            ok = False
        results.append((name, ok))

    print("=" * 64)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"自检汇总：{passed} 项 PASS，{failed} 项 FAIL")
    if failed:
        print("存在未通过项，请先修复上述 FAIL 后再启动 caption 生成任务。")
        return 1
    print("全部检查通过，环境就绪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
