"""Prompt definitions and response parsing for ReID caption generation.

本模块定义行人重识别（ReID）caption 生成所用的系统/用户提示词，
并提供对 VLM 原始回复的解析函数。所有提示词均为英文，以适配
Qwen3-VL 系列模型的英文指令遵循习惯。

约定（与 generate_captions.py 及下游质量过滤脚本共享）：
- VLM 被要求恰好输出 6 行，每行一个固定主题；
- 看不清的主题必须写 'not visible'，解析时整行丢弃；
- 每行最终被截断为一句短描述，作为 captions 列表的一个元素。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are an expert annotation assistant for person re-identification "
    "(ReID). You describe the *person* in surveillance-style images with "
    "short, factual, visually grounded phrases. You never guess: if an "
    "attribute is not clearly visible, you say 'not visible'. You never "
    "describe or imagine anything that is not in the image."
)

USER_PROMPT: str = """Describe the person in this image for a person re-identification (ReID) system.

Output EXACTLY 6 lines, one line per topic, in this fixed order:
1. Upper-body clothing: color and style (e.g. jacket, t-shirt, hoodie)
2. Lower-body clothing and shoes: color and style (e.g. jeans, skirt, sneakers)
3. Gender, approximate age group, and hair (e.g. male, adult, short black hair)
4. Carried items and accessories (e.g. backpack, handbag, hat, glasses)
5. Action, pose, and facing direction (e.g. walking, standing, facing left)
6. Scene: indoor/outdoor and lighting (e.g. outdoor, daylight)

Rules:
- One short phrase per line, at most 15 words per line.
- No numbering, no bullet symbols, no topic labels — just the description text.
- Use common color words only (e.g. black, white, gray, red, blue, green, yellow, pink, purple, brown, orange, beige, navy, khaki).
- If a topic is not clearly visible in the image, write exactly: not visible
- Describe only what you actually see. Never guess, never infer, never imagine anything that is not in the image.
- Focus on the main person if several people appear."""

# 常用颜色白名单：供下游质量过滤/归一化脚本使用（本模块不强制校验）。
COLOR_WHITELIST: tuple[str, ...] = (
    "black",
    "white",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "purple",
    "brown",
    "orange",
    "beige",
    "navy",
    "khaki",
    "silver",
    "gold",
    "dark blue",
    "light blue",
)

# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

# 行首序号/符号前缀：'1.' '1)' '1、' '- ' '* ' '①' 等
_PREFIX_RE = re.compile(
    r"^\s*(?:\d+\s*[.)、:：]?\s*|[-*•]\s*|[①②③④⑤⑥]\s*)+"
)

# 主题标签前缀（模型偶尔不遵守规则时会带上，如 'Upper body: ...'）
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z /&-]{0,40}:\s*")

_NOT_VISIBLE_RE = re.compile(r"not\s+visible", re.IGNORECASE)


def build_prompt() -> str:
    """返回完整的用户提示词（图片由 vLLM 多模态消息单独传入）。"""
    return USER_PROMPT


def parse_response(text: str) -> list[str]:
    """把 VLM 原始回复解析成干净的短句列表。

    处理规则：
    1. 按行切分，剥掉行首序号/符号前缀与可能的主题标签；
    2. 丢弃空行以及含 'not visible'（大小写不敏感）的行；
    3. 每行截到第一个句号（'.'）之前，去掉尾部空白；
    4. 二次过滤截断后为空的内容。

    Args:
        text: VLM 的原始文本回复。

    Returns:
        干净短句列表，顺序与原回复一致；数量可能少于 6（主题不可见时）。
    """
    captions: list[str] = []
    if not text:
        return captions

    for raw_line in text.splitlines():
        line = _PREFIX_RE.sub("", raw_line).strip()
        line = _LABEL_RE.sub("", line).strip()
        if not line:
            continue
        if _NOT_VISIBLE_RE.search(line):
            continue
        # 截到第一个句号前，保证每条是一句短描述
        line = line.split(".", 1)[0].strip()
        if line:
            captions.append(line)

    return captions


__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "COLOR_WHITELIST",
    "build_prompt",
    "parse_response",
]
