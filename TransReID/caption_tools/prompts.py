"""Versioned prompts and response parsing for ReID caption generation.

Prompt V1 is retained verbatim for reproducibility and ablation studies.
Prompt V2 extracts structured, identity-relevant visual attributes and renders
them deterministically into compact natural-language captions for CLIP/SigLIP.
Prompt V2.1 keeps the V2 schema but requires complete item phrases and removes
base-garment restatement from distinctive features.
Prompt V2.2 adds confidence-aware footwear naming and emits one canonical
training caption instead of overlapping caption views.
Prompt V2.3 keeps useful generic graphics while prioritizing identifiable
feature categories and requiring discriminative graphic qualifiers.
Prompt V2.4 reserves logo/text/patch for visually supported semantics and uses
mark as the conservative fallback for a visible but unclassifiable compact mark.
"""

from __future__ import annotations

import json
import re
from typing import Any

PROMPT_V1 = "v1"
PROMPT_V2 = "v2"
PROMPT_V2_1 = "v2.1"
PROMPT_V2_2 = "v2.2"
PROMPT_V2_3 = "v2.3"
PROMPT_V2_4 = "v2.4"
DEFAULT_PROMPT_VERSION = PROMPT_V2_4
SUPPORTED_PROMPT_VERSIONS: tuple[str, ...] = (
    PROMPT_V1,
    PROMPT_V2,
    PROMPT_V2_1,
    PROMPT_V2_2,
    PROMPT_V2_3,
    PROMPT_V2_4,
)

# ---------------------------------------------------------------------------
# Prompt V1 — frozen baseline
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V1: str = (
    "You are an expert annotation assistant for person re-identification "
    "(ReID). You describe the *person* in surveillance-style images with "
    "short, factual, visually grounded phrases. You never guess: if an "
    "attribute is not clearly visible, you say 'not visible'. You never "
    "describe or imagine anything that is not in the image."
)

USER_PROMPT_V1: str = """Describe the person in this image for a person re-identification (ReID) system.

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

# Backward-compatible aliases used by older imports.
SYSTEM_PROMPT = SYSTEM_PROMPT_V1
USER_PROMPT = USER_PROMPT_V1

# ---------------------------------------------------------------------------
# Prompt V2 — structured ReID attributes
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V2: str = """You are a strict visual attribute extractor for person re-identification.

Extract only clearly visible, identity-relevant appearance attributes of the
main person. Do not infer hidden attributes.

Ignore:
- gender, age, ethnicity, and facial identity
- action, pose, and facing direction
- background, scene, camera, weather, and lighting
- image quality or capture conditions

Use positive visual evidence only. Never claim that an item is absent.
If an attribute is unclear, occluded, or ambiguous, omit it.
Never use alternatives such as "black or blue", and never use words such as
"possibly", "maybe", "appears", or "likely".

Return valid JSON only."""

USER_PROMPT_V2: str = """Extract the visible ReID appearance attributes of the main person.

Return exactly one JSON object with this schema:

{
  "upper_clothing": {
    "visibility": "clear|partial|not_visible",
    "attributes": []
  },
  "lower_clothing": {
    "visibility": "clear|partial|not_visible",
    "attributes": []
  },
  "footwear": {
    "visibility": "clear|partial|not_visible",
    "attributes": []
  },
  "carried_items": [],
  "accessories": [],
  "hair": [],
  "distinctive_features": []
}

Rules:
- Each attribute must be a short lowercase phrase.
- Use coarse common color words.
- Include garment type, sleeve or length, material, and pattern only when clear.
- Put logos, stripes, large graphics, color blocks, and unusual patterns in distinctive_features.
- Use an empty list when no reliable positive attribute can be extracted.
- Do not write natural-language sentences outside the JSON object.
- Do not describe gender, age, pose, direction, scene, background, or lighting.
- Do not repeat the same information in multiple fields."""

# ---------------------------------------------------------------------------
# Prompt V2.1 — complete noun-headed item phrases
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V2_1: str = """You are a strict visual attribute extractor for person re-identification.

Extract only clearly visible, identity-relevant appearance attributes of the
main person. Do not infer hidden attributes.

Ignore:
- gender, age, ethnicity, and facial identity
- action, pose, and facing direction
- background, scene, camera, weather, and lighting
- image quality or capture conditions

Use positive visual evidence only. Never claim that an item is absent.
If an attribute is unclear, occluded, or ambiguous, omit it.
Never use alternatives such as "black or blue", and never use words such as
"possibly", "maybe", "appears", or "likely".

Every list element must be a complete noun-headed description of one visible
item or one distinctive mark. Never split one item into atomic properties.

Return valid JSON only."""

USER_PROMPT_V2_1: str = """Extract the visible ReID appearance attributes of the main person.

Return exactly one JSON object with this schema:

{
  "upper_clothing": {
    "visibility": "clear|partial|not_visible",
    "attributes": []
  },
  "lower_clothing": {
    "visibility": "clear|partial|not_visible",
    "attributes": []
  },
  "footwear": {
    "visibility": "clear|partial|not_visible",
    "attributes": []
  },
  "carried_items": [],
  "accessories": [],
  "hair": [],
  "distinctive_features": []
}

Rules:
- Each element in upper_clothing, lower_clothing, and footwear describes one
  complete visible item, never one isolated property.
- Combine every reliable property of the same item into one noun-headed phrase:
  color + material or pattern + sleeve, length, or style + item type.
- Never output standalone properties such as "red", "dark", "short sleeve",
  "knee-length", "full length", "flat sole", or "loose fit".
- Good: ["red short-sleeved t-shirt"]
  Bad: ["red", "short sleeve", "t-shirt"]
- Good: ["dark blue knee-length shorts"]
  Bad: ["shorts", "dark blue", "knee-length"]
- Good: ["black sandals"]
  Bad: ["black", "sandals"]
- Hair must be a complete noun phrase such as "short dark hair" or
  "black ponytail", never "short", "dark", or "black".
- Each carried item and accessory must name the visible item.
- Put logos, stripes, large graphics, color blocks, and unusual patterns in
  distinctive_features.
- A distinctive feature describes only the discriminative mark, optionally its
  own color and location. Do not repeat the base garment type or base garment
  color already recorded in clothing fields.
- Good:
  upper_clothing: ["white t-shirt"]
  distinctive_features: ["large black front graphic"]
- Bad:
  distinctive_features: ["large black graphic on white t-shirt"]
- Use location words such as "front", "chest", "back", or "side" instead of
  repeating the garment.
- Use coarse common color words and lowercase phrases.
- Use an empty list when no reliable positive attribute can be extracted.
- Do not write natural-language sentences outside the JSON object.
- Do not describe gender, age, pose, direction, scene, background, or lighting.
- Do not repeat the same information in multiple fields."""

# ---------------------------------------------------------------------------
# Prompt V2.2 — confidence-aware footwear and stricter feature isolation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V2_2: str = SYSTEM_PROMPT_V2_1

USER_PROMPT_V2_2: str = USER_PROMPT_V2_1 + """

V2.2 hard constraints:
- For footwear, use the most specific type that is clearly visible:
  sandals, sneakers, boots, loafers, dress shoes, heels, slippers, or shoes.
- If the exact footwear type is unclear, use "[color] shoes".
- Never use the generic word "footwear".
- Never guess a specific footwear type from a blurred shape.
- distinctive_features must never contain a garment noun such as shirt,
  t-shirt, jacket, hoodie, pants, jeans, shorts, skirt, or dress.
- Describe feature location with "front", "chest", "back", or "side", without
  naming the garment.
- Good: "large black front graphic"
- Bad: "large black graphic on white t-shirt"
"""

# ---------------------------------------------------------------------------
# Prompt V2.3 — useful but non-redundant graphic descriptions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V2_3: str = SYSTEM_PROMPT_V2_2

USER_PROMPT_V2_3: str = USER_PROMPT_V2_2 + """

V2.3 distinctive-feature taxonomy:
- Prefer a specific visible category over the generic word "graphic":
  logo or emblem, text or lettering, stripes, color block, patch, trim, print.
- Use "logo" only when the mark is clearly logo-like or emblem-like.
- Use "text" or "lettering" only when text-like shapes are clearly visible.
- Use "graphic" only for a discrete pictorial or printed design whose exact
  category or content cannot be identified.
- Every phrase containing "graphic" must include at least one discriminative
  color, size, or shape qualifier; also include location when visible.
- Good: "large black circular front graphic"
- Bad: "graphic" or "front graphic"
- Never guess the semantic content of a blurred graphic.
- Omit a feature when even the presence of a distinct mark is uncertain.
"""

# ---------------------------------------------------------------------------
# Prompt V2.4 — confidence-aware semantic labels for compact marks
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V2_4: str = SYSTEM_PROMPT_V2_3

USER_PROMPT_V2_4: str = USER_PROMPT_V2_3 + """

V2.4 semantic-confidence rules:
- Use "logo" only when the mark is clearly recognizable as a logo, emblem,
  badge, or brand-like symbol.
- Do not infer a logo from a small blurred or high-contrast blob.
- If a compact visible mark is present but its category is unclear, use "mark",
  for example "small white chest mark".
- Use "patch" only when a distinct attached or rectangular patch is visible.
- Use "text" or "lettering" only when letter-like shapes are clearly visible,
  even if the exact words cannot be read.
- Use "graphic" for a pictorial or printed design whose exact category cannot
  be identified.
"""

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

V2_REGION_KEYS: tuple[str, ...] = (
    "upper_clothing",
    "lower_clothing",
    "footwear",
)
V2_LIST_KEYS: tuple[str, ...] = (
    "carried_items",
    "accessories",
    "hair",
    "distinctive_features",
)
V2_VISIBILITY_VALUES: frozenset[str] = frozenset(
    {"clear", "partial", "not_visible"}
)

# ---------------------------------------------------------------------------
# Version selection
# ---------------------------------------------------------------------------


def normalize_prompt_version(version: str | None) -> str:
    normalized = (version or DEFAULT_PROMPT_VERSION).strip().lower()
    if normalized not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(
            f"unsupported prompt version {version!r}; "
            f"expected one of {SUPPORTED_PROMPT_VERSIONS}"
        )
    return normalized


def get_system_prompt(version: str | None = None) -> str:
    version = normalize_prompt_version(version)
    if version == PROMPT_V1:
        return SYSTEM_PROMPT_V1
    if version == PROMPT_V2:
        return SYSTEM_PROMPT_V2
    if version == PROMPT_V2_1:
        return SYSTEM_PROMPT_V2_1
    if version == PROMPT_V2_2:
        return SYSTEM_PROMPT_V2_2
    if version == PROMPT_V2_3:
        return SYSTEM_PROMPT_V2_3
    return SYSTEM_PROMPT_V2_4


def build_prompt(version: str | None = None) -> str:
    version = normalize_prompt_version(version)
    if version == PROMPT_V1:
        return USER_PROMPT_V1
    if version == PROMPT_V2:
        return USER_PROMPT_V2
    if version == PROMPT_V2_1:
        return USER_PROMPT_V2_1
    if version == PROMPT_V2_2:
        return USER_PROMPT_V2_2
    if version == PROMPT_V2_3:
        return USER_PROMPT_V2_3
    return USER_PROMPT_V2_4


# ---------------------------------------------------------------------------
# V1 parsing
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"^\s*(?:\d+\s*[.)、:：]?\s*|[-*•]\s*|[①②③④⑤⑥]\s*)+"
)
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z /&-]{0,40}:\s*")
_NOT_VISIBLE_RE = re.compile(r"not\s+visible", re.IGNORECASE)


def parse_v1_response(text: str) -> list[str]:
    captions: list[str] = []
    if not text:
        return captions

    for raw_line in text.splitlines():
        line = _PREFIX_RE.sub("", raw_line).strip()
        line = _LABEL_RE.sub("", line).strip()
        if not line or _NOT_VISIBLE_RE.search(line):
            continue
        line = line.split(".", 1)[0].strip()
        if line:
            captions.append(line)
    return captions


# ---------------------------------------------------------------------------
# V2 parsing and deterministic caption rendering
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL
)
_SPACE_RE = re.compile(r"\s+")
_UNCERTAIN_RE = re.compile(
    r"\b(?:possibly|maybe|appears?|likely|seems?|perhaps|apparently)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_RE = re.compile(r"\b\w+\s+or\s+\w+\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(?:"
    r"male|female|man|woman|boy|girl|adult|child|teen|elderly|"
    r"walking|standing|running|sitting|facing|"
    r"indoor|outdoor|daylight|lighting|background|camera"
    r")\b",
    re.IGNORECASE,
)
_V2_1_ITEM_PATTERNS: dict[str, re.Pattern[str]] = {
    "upper_clothing": re.compile(
        r"\b(?:t-?shirt|shirt|jacket|hoodie|coat|sweater|top|blouse|vest|"
        r"jersey|dress|uniform|cardigan|pullover)\b",
        re.IGNORECASE,
    ),
    "lower_clothing": re.compile(
        r"\b(?:pants|trousers|jeans|shorts|skirt|leggings|dress|overalls|"
        r"sweatpants|joggers)\b",
        re.IGNORECASE,
    ),
    "footwear": re.compile(
        r"\b(?:shoe|shoes|sneaker|sneakers|sandal|sandals|boot|boots|"
        r"slipper|slippers|loafer|loafers|heel|heels|footwear)\b",
        re.IGNORECASE,
    ),
    "carried_items": re.compile(
        r"\b(?:backpack|bag|handbag|purse|briefcase|suitcase|umbrella|"
        r"phone|bottle|book|folder|box|parcel|object)\b",
        re.IGNORECASE,
    ),
    "accessories": re.compile(
        r"\b(?:cap|hat|glasses|eyeglasses|sunglasses|belt|watch|bracelet|"
        r"necklace|scarf|tie|mask|headband|earring|earrings)\b",
        re.IGNORECASE,
    ),
    "hair": re.compile(
        r"\b(?:hair|ponytail|braid|braids|bun|dreadlocks|hairstyle)\b",
        re.IGNORECASE,
    ),
}
_V2_1_FEATURE_RE = re.compile(
    r"\b(?:graphic|graphics|logo|logos|stripe|stripes|pattern|print|design|"
    r"lettering|text|patch|emblem|mark|motif|color block|character|outline)\b",
    re.IGNORECASE,
)
_GARMENT_NOUNS = (
    r"t-?shirt|shirt|jacket|hoodie|coat|sweater|top|blouse|vest|jersey|"
    r"dress|uniform|cardigan|pullover|pants|trousers|jeans|shorts|skirt|"
    r"leggings|overalls|sweatpants|joggers"
)
_GARMENT_CONTEXT_SUFFIX_RE = re.compile(
    rf"\s+on\s+(?:the\s+)?"
    rf"(?:(?:(?:left|right|center|central)\s+)?"
    rf"(?:front|back|side|chest)\s+(?:of\s+)?)?"
    rf"(?:[\w-]+\s+){{0,3}}(?:{_GARMENT_NOUNS})$",
    re.IGNORECASE,
)
_GARMENT_WITH_FEATURE_RE = re.compile(
    rf"^(?:[\w-]+\s+){{0,3}}(?:{_GARMENT_NOUNS})\s+with\s+(.+)$",
    re.IGNORECASE,
)
_GRAPHIC_SPECIFICITY_RE = re.compile(
    r"\b(?:"
    r"black|white|gray|red|blue|green|yellow|pink|purple|brown|orange|"
    r"beige|navy|khaki|silver|gold|dark|light|multicolor|high-contrast|"
    r"large|small|circular|round|rectangular|square"
    r")\b",
    re.IGNORECASE,
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    fenced = _CODE_FENCE_RE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0:
        return None
    value = None
    if end > start:
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            value = None
    if isinstance(value, dict):
        return value

    # A rare deterministic-generation failure can exhaust max_new_tokens
    # while enumerating the final distinctive_features list. Earlier fields
    # are still complete and useful. Repair only this exact schema-tail case;
    # discard the incomplete final field instead of guessing its contents.
    marker = '"distinctive_features"'
    marker_pos = stripped.find(marker, start)
    if marker_pos >= 0:
        candidate = stripped[start:marker_pos] + '"distinctive_features": []}'
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = None
    return value if isinstance(value, dict) else None


def _normalize_attribute(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    attribute = _SPACE_RE.sub(" ", value.strip().lower()).strip(" .,:;-")
    if not attribute or len(attribute.split()) > 12:
        return None
    if (
        _NOT_VISIBLE_RE.search(attribute)
        or _UNCERTAIN_RE.search(attribute)
        or _ALTERNATIVE_RE.search(attribute)
        or _FORBIDDEN_RE.search(attribute)
    ):
        return None
    return attribute


def _normalize_attribute_list(value: Any, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        attribute = _normalize_attribute(item)
        if attribute is None or attribute in seen:
            continue
        normalized.append(attribute)
        seen.add(attribute)
        if len(normalized) >= limit:
            break
    return normalized


def parse_v2_attributes(text: str) -> dict[str, Any] | None:
    payload = _extract_json_object(text)
    if payload is None:
        return None

    result: dict[str, Any] = {}
    for key in V2_REGION_KEYS:
        region = payload.get(key)
        if not isinstance(region, dict):
            region = {}
        visibility = str(region.get("visibility", "not_visible")).strip().lower()
        if visibility not in V2_VISIBILITY_VALUES:
            visibility = "not_visible"
        attributes = _normalize_attribute_list(region.get("attributes"))
        if not attributes:
            visibility = "not_visible"
        result[key] = {
            "visibility": visibility,
            "attributes": attributes,
        }

    for key in V2_LIST_KEYS:
        result[key] = _normalize_attribute_list(payload.get(key))

    return result


def _normalize_v2_1_item_list(value: Any, field: str) -> list[str]:
    values = _normalize_attribute_list(value)
    pattern = _V2_1_ITEM_PATTERNS[field]
    return [attribute for attribute in values if pattern.search(attribute)]


def _normalize_v2_1_distinctive_list(value: Any) -> list[str]:
    values = _normalize_attribute_list(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for attribute in values:
        garment_with_feature = _GARMENT_WITH_FEATURE_RE.match(attribute)
        if garment_with_feature:
            attribute = garment_with_feature.group(1).strip()
        attribute = _GARMENT_CONTEXT_SUFFIX_RE.sub("", attribute).strip()
        if (
            not attribute
            or not _V2_1_FEATURE_RE.search(attribute)
            or attribute in seen
        ):
            continue
        normalized.append(attribute)
        seen.add(attribute)
    return normalized


def parse_v2_1_attributes(text: str) -> dict[str, Any] | None:
    payload = _extract_json_object(text)
    if payload is None:
        return None

    result: dict[str, Any] = {}
    for key in V2_REGION_KEYS:
        region = payload.get(key)
        if not isinstance(region, dict):
            region = {}
        visibility = str(region.get("visibility", "not_visible")).strip().lower()
        if visibility not in V2_VISIBILITY_VALUES:
            visibility = "not_visible"
        attributes = _normalize_v2_1_item_list(region.get("attributes"), key)
        if not attributes:
            visibility = "not_visible"
        result[key] = {
            "visibility": visibility,
            "attributes": attributes,
        }

    for key in ("carried_items", "accessories", "hair"):
        result[key] = _normalize_v2_1_item_list(payload.get(key), key)
    result["distinctive_features"] = _normalize_v2_1_distinctive_list(
        payload.get("distinctive_features")
    )
    return result


def parse_v2_2_attributes(text: str) -> dict[str, Any] | None:
    attributes = parse_v2_1_attributes(text)
    if attributes is None:
        return None
    footwear = attributes["footwear"]
    footwear["attributes"] = [
        attribute
        for attribute in footwear["attributes"]
        if not re.search(r"\bfootwear\b", attribute, re.IGNORECASE)
    ]
    if not footwear["attributes"]:
        footwear["visibility"] = "not_visible"
    return attributes


def parse_v2_3_attributes(text: str) -> dict[str, Any] | None:
    attributes = parse_v2_2_attributes(text)
    if attributes is None:
        return None
    attributes["distinctive_features"] = [
        feature
        for feature in attributes["distinctive_features"]
        if (
            not re.search(r"\bgraphic\b", feature, re.IGNORECASE)
            or _GRAPHIC_SPECIFICITY_RE.search(feature)
        )
    ]
    return attributes


def parse_v2_4_attributes(text: str) -> dict[str, Any] | None:
    """V2.4 shares V2.3 structural validation.

    Whether a visible mark is truly a logo, text, patch, graphic, or generic
    mark depends on image evidence and therefore belongs in the vision prompt,
    not in text-only postprocessing.
    """
    return parse_v2_3_attributes(text)


def _join_phrases(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def render_v2_captions(attributes: dict[str, Any]) -> list[str]:
    clothing: list[str] = []
    for key in V2_REGION_KEYS:
        region = attributes.get(key, {})
        if isinstance(region, dict):
            clothing.extend(_normalize_attribute_list(region.get("attributes")))

    carried = _normalize_attribute_list(attributes.get("carried_items"))
    accessories = _normalize_attribute_list(attributes.get("accessories"))
    hair = _normalize_attribute_list(attributes.get("hair"))
    distinctive = _normalize_attribute_list(attributes.get("distinctive_features"))

    clauses: list[str] = []
    if clothing:
        clauses.append(f"wearing {_join_phrases(clothing)}")
    if carried:
        clauses.append(f"carrying {_join_phrases(carried)}")
    details = accessories + hair + distinctive
    if details:
        clauses.append(f"with {_join_phrases(details)}")

    captions: list[str] = []
    if clauses:
        captions.append(f"a person {', '.join(clauses)}")
    if clothing:
        captions.append(f"a person wearing {_join_phrases(clothing)}")
    if distinctive:
        captions.append(f"a person with {_join_phrases(distinctive)}")

    return list(dict.fromkeys(captions))


def render_v2_2_captions(attributes: dict[str, Any]) -> list[str]:
    """Return one canonical full-description caption for training."""
    captions = render_v2_captions(attributes)
    # Preserve the one-image/one-contract-record invariant for severely
    # occluded or unusable crops. The downstream quality score remains 0.0
    # when every reliable attribute is empty, so training can filter or
    # down-weight this deliberately generic fallback.
    return captions[:1] or ["a person"]


def parse_response_payload(
    text: str,
    version: str | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    version = normalize_prompt_version(version)
    if version == PROMPT_V1:
        return parse_v1_response(text), None
    if version == PROMPT_V2:
        attributes = parse_v2_attributes(text)
    elif version == PROMPT_V2_1:
        attributes = parse_v2_1_attributes(text)
    elif version == PROMPT_V2_2:
        attributes = parse_v2_2_attributes(text)
    elif version == PROMPT_V2_3:
        attributes = parse_v2_3_attributes(text)
    else:
        attributes = parse_v2_4_attributes(text)
    if attributes is None:
        return [], None
    captions = (
        render_v2_2_captions(attributes)
        if version in (PROMPT_V2_2, PROMPT_V2_3, PROMPT_V2_4)
        else render_v2_captions(attributes)
    )
    return captions, attributes


def parse_response(text: str, version: str | None = PROMPT_V1) -> list[str]:
    """Backward-compatible parser returning only rendered caption strings."""
    captions, _ = parse_response_payload(text, version)
    return captions


__all__ = [
    "COLOR_WHITELIST",
    "DEFAULT_PROMPT_VERSION",
    "PROMPT_V1",
    "PROMPT_V2",
    "PROMPT_V2_1",
    "PROMPT_V2_2",
    "PROMPT_V2_3",
    "PROMPT_V2_4",
    "SUPPORTED_PROMPT_VERSIONS",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_V1",
    "SYSTEM_PROMPT_V2",
    "SYSTEM_PROMPT_V2_1",
    "SYSTEM_PROMPT_V2_2",
    "SYSTEM_PROMPT_V2_3",
    "SYSTEM_PROMPT_V2_4",
    "USER_PROMPT",
    "USER_PROMPT_V1",
    "USER_PROMPT_V2",
    "USER_PROMPT_V2_1",
    "USER_PROMPT_V2_2",
    "USER_PROMPT_V2_3",
    "USER_PROMPT_V2_4",
    "build_prompt",
    "get_system_prompt",
    "normalize_prompt_version",
    "parse_response",
    "parse_response_payload",
    "parse_v1_response",
    "parse_v2_attributes",
    "parse_v2_1_attributes",
    "parse_v2_2_attributes",
    "parse_v2_3_attributes",
    "parse_v2_4_attributes",
    "render_v2_captions",
    "render_v2_2_captions",
]
