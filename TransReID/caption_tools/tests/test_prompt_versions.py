from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CAPTION_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPTION_TOOLS))

from generate_captions import load_done_paths, parse_args  # noqa: E402
from export_jsonl import _iter_clean_records  # noqa: E402
from postprocess import process_record  # noqa: E402
from prompts import (  # noqa: E402
    PROMPT_V1,
    PROMPT_V2,
    PROMPT_V2_1,
    PROMPT_V2_2,
    PROMPT_V2_3,
    PROMPT_V2_4,
    build_prompt,
    parse_response,
    parse_response_payload,
    parse_v2_1_attributes,
    parse_v2_2_attributes,
    parse_v2_3_attributes,
    parse_v2_4_attributes,
    parse_v2_attributes,
)


class PromptVersionTests(unittest.TestCase):
    def test_v1_prompt_and_parser_remain_available(self) -> None:
        self.assertIn("EXACTLY 6 lines", build_prompt(PROMPT_V1))
        response = "\n".join(
            [
                "1. Red short-sleeve t-shirt.",
                "Blue denim shorts.",
                "not visible",
                "Black backpack.",
                "Walking, facing left.",
                "Outdoor, daylight.",
            ]
        )
        self.assertEqual(
            parse_response(response, PROMPT_V1),
            [
                "Red short-sleeve t-shirt",
                "Blue denim shorts",
                "Black backpack",
                "Walking, facing left",
                "Outdoor, daylight",
            ],
        )

    def test_v2_parses_json_and_renders_compact_captions(self) -> None:
        response = """```json
{
  "upper_clothing": {
    "visibility": "clear",
    "attributes": ["red short-sleeve t-shirt"]
  },
  "lower_clothing": {
    "visibility": "clear",
    "attributes": ["blue denim knee-length shorts"]
  },
  "footwear": {
    "visibility": "partial",
    "attributes": ["dark shoes"]
  },
  "carried_items": [],
  "accessories": ["black cap"],
  "hair": ["short dark hair"],
  "distinctive_features": ["circular front graphic"]
}
```"""
        captions, attributes = parse_response_payload(response, PROMPT_V2)
        self.assertIsNotNone(attributes)
        self.assertEqual(attributes["footwear"]["visibility"], "partial")
        self.assertGreaterEqual(len(captions), 2)
        self.assertTrue(captions[0].startswith("a person wearing"))
        self.assertIn("circular front graphic", captions[0])
        self.assertNotIn("outdoor", captions[0])

    def test_v2_drops_uncertain_and_forbidden_attributes(self) -> None:
        response = json.dumps(
            {
                "upper_clothing": {
                    "visibility": "clear",
                    "attributes": ["red or brown shirt", "red t-shirt"],
                },
                "lower_clothing": {
                    "visibility": "clear",
                    "attributes": ["possibly blue shorts"],
                },
                "footwear": {
                    "visibility": "partial",
                    "attributes": ["black shoes"],
                },
                "carried_items": [],
                "accessories": [],
                "hair": ["male adult", "short dark hair"],
                "distinctive_features": ["not visible"],
            }
        )
        attributes = parse_v2_attributes(response)
        self.assertIsNotNone(attributes)
        self.assertEqual(
            attributes["upper_clothing"]["attributes"],
            ["red t-shirt"],
        )
        self.assertEqual(
            attributes["lower_clothing"],
            {"visibility": "not_visible", "attributes": []},
        )
        self.assertEqual(attributes["hair"], ["short dark hair"])
        self.assertEqual(attributes["distinctive_features"], [])

    def test_v2_1_requires_complete_items_and_removes_garment_context(
        self,
    ) -> None:
        self.assertIn(
            "Never output standalone properties",
            build_prompt(PROMPT_V2_1),
        )
        response = json.dumps(
            {
                "upper_clothing": {
                    "visibility": "clear",
                    "attributes": [
                        "red",
                        "short sleeve",
                        "red short-sleeved t-shirt",
                    ],
                },
                "lower_clothing": {
                    "visibility": "clear",
                    "attributes": ["blue", "dark blue knee-length shorts"],
                },
                "footwear": {
                    "visibility": "partial",
                    "attributes": ["black", "black sandals"],
                },
                "carried_items": [],
                "accessories": [],
                "hair": ["short", "short dark hair"],
                "distinctive_features": [
                    "large black graphic on white t-shirt",
                    "white t-shirt",
                    "horizontal stripes on shirt",
                    "small logo on left chest of purple polo shirt",
                    "large black front graphic",
                ],
            }
        )
        attributes = parse_v2_1_attributes(response)
        self.assertIsNotNone(attributes)
        self.assertEqual(
            attributes["upper_clothing"]["attributes"],
            ["red short-sleeved t-shirt"],
        )
        self.assertEqual(
            attributes["lower_clothing"]["attributes"],
            ["dark blue knee-length shorts"],
        )
        self.assertEqual(
            attributes["footwear"]["attributes"],
            ["black sandals"],
        )
        self.assertEqual(attributes["hair"], ["short dark hair"])
        self.assertEqual(
            attributes["distinctive_features"],
            [
                "large black graphic",
                "horizontal stripes",
                "small logo",
                "large black front graphic",
            ],
        )

        captions, payload = parse_response_payload(response, PROMPT_V2_1)
        self.assertEqual(payload, attributes)
        self.assertTrue(captions)
        self.assertNotIn("on white t-shirt", " ".join(captions))

    def test_v2_2_uses_specific_footwear_and_one_canonical_caption(
        self,
    ) -> None:
        prompt = build_prompt(PROMPT_V2_2)
        self.assertIn('Never use the generic word "footwear"', prompt)
        self.assertIn("must never contain a garment noun", prompt)
        response = json.dumps(
            {
                "upper_clothing": {
                    "visibility": "clear",
                    "attributes": ["red short-sleeved t-shirt"],
                },
                "lower_clothing": {
                    "visibility": "clear",
                    "attributes": ["dark blue knee-length shorts"],
                },
                "footwear": {
                    "visibility": "partial",
                    "attributes": ["dark footwear", "black sandals"],
                },
                "carried_items": [],
                "accessories": [],
                "hair": ["short dark hair"],
                "distinctive_features": [
                    "large black graphic on white t-shirt"
                ],
            }
        )
        attributes = parse_v2_2_attributes(response)
        self.assertIsNotNone(attributes)
        self.assertEqual(
            attributes["footwear"]["attributes"],
            ["black sandals"],
        )
        captions, payload = parse_response_payload(response, PROMPT_V2_2)
        self.assertEqual(payload, attributes)
        self.assertEqual(len(captions), 1)
        self.assertIn("black sandals", captions[0])
        self.assertIn("large black graphic", captions[0])

        raw = {
            "dataset": "market1501",
            "split": "train",
            "image_path": "bounding_box_train/v2_2.jpg",
            "pid": 2,
            "raw_response": response,
            "generator": "test-model",
            "prompt_version": PROMPT_V2_2,
        }
        record = process_record(raw, min_captions=2, dedup_threshold=0.9)
        self.assertIsNotNone(record)
        self.assertEqual(record["captions"], captions)
        self.assertEqual(record["postprocess_version"], "p2")
        self.assertEqual(record["renderer_version"], "r2-canonical")

    def test_v2_3_keeps_specific_graphics_and_drops_generic_graphics(
        self,
    ) -> None:
        prompt = build_prompt(PROMPT_V2_3)
        self.assertIn(
            'Use "graphic" only for a discrete pictorial or printed design',
            prompt,
        )
        response = json.dumps(
            {
                "upper_clothing": {
                    "visibility": "clear",
                    "attributes": ["white short-sleeved t-shirt"],
                },
                "lower_clothing": {
                    "visibility": "clear",
                    "attributes": ["dark blue jeans"],
                },
                "footwear": {
                    "visibility": "partial",
                    "attributes": ["dark shoes"],
                },
                "carried_items": [],
                "accessories": [],
                "hair": ["short dark hair"],
                "distinctive_features": [
                    "graphic",
                    "front graphic",
                    "large black circular front graphic",
                    "small white chest logo",
                    "horizontal red stripes",
                ],
            }
        )
        attributes = parse_v2_3_attributes(response)
        self.assertIsNotNone(attributes)
        self.assertEqual(
            attributes["distinctive_features"],
            [
                "large black circular front graphic",
                "small white chest logo",
                "horizontal red stripes",
            ],
        )
        captions, payload = parse_response_payload(response, PROMPT_V2_3)
        self.assertEqual(payload, attributes)
        self.assertEqual(len(captions), 1)

    def test_v2_4_uses_mark_without_rewriting_supported_feature_types(
        self,
    ) -> None:
        prompt = build_prompt(PROMPT_V2_4)
        self.assertIn(
            'If a compact visible mark is present but its category is unclear, use "mark"',
            prompt,
        )
        response = json.dumps(
            {
                "upper_clothing": {
                    "visibility": "clear",
                    "attributes": ["white short-sleeved t-shirt"],
                },
                "lower_clothing": {
                    "visibility": "clear",
                    "attributes": ["dark blue jeans"],
                },
                "footwear": {
                    "visibility": "partial",
                    "attributes": ["dark shoes"],
                },
                "carried_items": [],
                "accessories": [],
                "hair": ["short dark hair"],
                "distinctive_features": [
                    "small white chest mark",
                    "large black front text",
                    "large circular front graphic",
                ],
            }
        )
        attributes = parse_v2_4_attributes(response)
        self.assertIsNotNone(attributes)
        self.assertEqual(
            attributes["distinctive_features"],
            [
                "small white chest mark",
                "large black front text",
                "large circular front graphic",
            ],
        )
        captions, payload = parse_response_payload(response, PROMPT_V2_4)
        self.assertEqual(payload, attributes)
        self.assertEqual(len(captions), 1)

    def test_v2_postprocess_keeps_attributes_and_version(self) -> None:
        raw = {
            "dataset": "market1501",
            "split": "train",
            "image_path": "bounding_box_train/0002.jpg",
            "pid": 2,
            "raw_response": json.dumps(
                {
                    "upper_clothing": {
                        "visibility": "clear",
                        "attributes": ["red t-shirt"],
                    },
                    "lower_clothing": {
                        "visibility": "clear",
                        "attributes": ["blue denim shorts"],
                    },
                    "footwear": {
                        "visibility": "partial",
                        "attributes": ["dark shoes"],
                    },
                    "carried_items": [],
                    "accessories": [],
                    "hair": ["short dark hair"],
                    "distinctive_features": ["circular front graphic"],
                }
            ),
            "generator": "test-model",
            "prompt_version": PROMPT_V2,
        }
        record = process_record(raw, min_captions=2, dedup_threshold=0.9)
        self.assertIsNotNone(record)
        self.assertEqual(record["prompt_version"], PROMPT_V2)
        self.assertIn("attributes", record)
        self.assertGreater(record["quality_score"], 0)
        self.assertTrue(record["captions"])

    def test_v2_4_postprocess_preserves_empty_attribute_record(self) -> None:
        raw = {
            "dataset": "market1501",
            "split": "train",
            "image_path": "bounding_box_train/0081.jpg",
            "pid": 81,
            "raw_response": json.dumps(
                {
                    "upper_clothing": {
                        "visibility": "not_visible",
                        "attributes": [],
                    },
                    "lower_clothing": {
                        "visibility": "not_visible",
                        "attributes": [],
                    },
                    "footwear": {
                        "visibility": "not_visible",
                        "attributes": [],
                    },
                    "carried_items": [],
                    "accessories": [],
                    "hair": [],
                    "distinctive_features": [],
                }
            ),
            "generator": "test-model",
            "prompt_version": PROMPT_V2_4,
        }
        record = process_record(raw, min_captions=2, dedup_threshold=0.9)
        self.assertIsNotNone(record)
        self.assertEqual(record["captions"], ["a person"])
        self.assertEqual(record["quality_score"], 0.0)
        self.assertEqual(record["prompt_version"], PROMPT_V2_4)
        self.assertEqual(record["renderer_version"], "r2.1-canonical-fallback")
        self.assertIn("attributes", record)

    def test_v2_4_repairs_truncated_final_distinctive_features(self) -> None:
        response = """{
          "upper_clothing": {
            "visibility": "clear",
            "attributes": ["white puffer jacket"]
          },
          "lower_clothing": {
            "visibility": "clear",
            "attributes": ["black skirt", "black leggings"]
          },
          "footwear": {
            "visibility": "clear",
            "attributes": ["black boots"]
          },
          "carried_items": ["black shoulder bag"],
          "accessories": [],
          "hair": [],
          "distinctive_features": [
            "white fur trim on hood",
            "white front panel on jacket"""
        captions, attributes = parse_response_payload(response, PROMPT_V2_4)
        self.assertEqual(len(captions), 1)
        self.assertIn("white puffer jacket", captions[0])
        self.assertIsNotNone(attributes)
        self.assertEqual(attributes["distinctive_features"], [])

    def test_resume_rejects_prompt_version_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "market1501.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "image_path": "bounding_box_train/0002.jpg",
                        "prompt_version": PROMPT_V1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_done_paths(output, PROMPT_V1),
                {"bounding_box_train/0002.jpg"},
            )
            with self.assertRaisesRegex(ValueError, "独立输出目录"):
                load_done_paths(output, PROMPT_V2)

    def test_default_output_directory_follows_prompt_version(self) -> None:
        args_default = parse_args(["--data-root", "/tmp/data"])
        args_v1 = parse_args(
            ["--data-root", "/tmp/data", "--prompt-version", PROMPT_V1]
        )
        args_v2 = parse_args(
            ["--data-root", "/tmp/data", "--prompt-version", PROMPT_V2]
        )
        args_v2_1 = parse_args(
            ["--data-root", "/tmp/data", "--prompt-version", PROMPT_V2_1]
        )
        args_v2_2 = parse_args(
            ["--data-root", "/tmp/data", "--prompt-version", PROMPT_V2_2]
        )
        args_v2_3 = parse_args(
            ["--data-root", "/tmp/data", "--prompt-version", PROMPT_V2_3]
        )
        args_v2_4 = parse_args(
            ["--data-root", "/tmp/data", "--prompt-version", PROMPT_V2_4]
        )
        self.assertEqual(args_default.prompt_version, PROMPT_V2_4)
        self.assertEqual(Path(args_default.output_dir).name, PROMPT_V2_4)
        self.assertEqual(Path(args_v1.output_dir).name, PROMPT_V1)
        self.assertEqual(Path(args_v2.output_dir).name, PROMPT_V2)
        self.assertEqual(Path(args_v2_1.output_dir).name, PROMPT_V2_1)
        self.assertEqual(Path(args_v2_2.output_dir).name, PROMPT_V2_2)
        self.assertEqual(Path(args_v2_3.output_dir).name, PROMPT_V2_3)
        self.assertEqual(Path(args_v2_4.output_dir).name, PROMPT_V2_4)

    def test_export_accepts_v2_and_defaults_legacy_records_to_v1(self) -> None:
        base = {
            "dataset": "market1501",
            "split": "train",
            "pid": 2,
            "captions": ["a person wearing a red t-shirt"],
            "quality_score": 0.8,
            "generator": "test-model",
        }
        with tempfile.TemporaryDirectory() as tmp:
            clean_dir = Path(tmp)
            legacy = {
                **base,
                "image_path": "bounding_box_train/legacy.jpg",
            }
            v2 = {
                **base,
                "image_path": "bounding_box_train/v2.jpg",
                "prompt_version": PROMPT_V2,
                "attributes": {
                    "upper_clothing": {
                        "visibility": "clear",
                        "attributes": ["red t-shirt"],
                    }
                },
            }
            v2_1 = {
                **base,
                "image_path": "bounding_box_train/v2_1.jpg",
                "prompt_version": PROMPT_V2_1,
                "attributes": {
                    "upper_clothing": {
                        "visibility": "clear",
                        "attributes": ["red t-shirt"],
                    }
                },
            }
            v2_2 = {
                **base,
                "image_path": "bounding_box_train/v2_2.jpg",
                "prompt_version": PROMPT_V2_2,
                "postprocess_version": "p2",
                "renderer_version": "r2-canonical",
                "attributes": {
                    "upper_clothing": {
                        "visibility": "clear",
                        "attributes": ["red t-shirt"],
                    }
                },
            }
            v2_3 = {
                **base,
                "image_path": "bounding_box_train/v2_3.jpg",
                "prompt_version": PROMPT_V2_3,
                "postprocess_version": "p2",
                "renderer_version": "r2-canonical",
                "attributes": {
                    "upper_clothing": {
                        "visibility": "clear",
                        "attributes": ["red t-shirt"],
                    }
                },
            }
            v2_4 = {
                **base,
                "image_path": "bounding_box_train/v2_4.jpg",
                "prompt_version": PROMPT_V2_4,
                "postprocess_version": "p2",
                "renderer_version": "r2-canonical",
                "attributes": {
                    "upper_clothing": {
                        "visibility": "clear",
                        "attributes": ["red t-shirt"],
                    }
                },
            }
            (clean_dir / "market1501.jsonl").write_text(
                "\n".join(
                    (
                        json.dumps(legacy),
                        json.dumps(v2),
                        json.dumps(v2_1),
                        json.dumps(v2_2),
                        json.dumps(v2_3),
                        json.dumps(v2_4),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            records, bad_lines, incomplete = _iter_clean_records(clean_dir)

        self.assertEqual((bad_lines, incomplete), (0, 0))
        self.assertEqual(
            [record["prompt_version"] for record in records],
            [
                PROMPT_V1,
                PROMPT_V2,
                PROMPT_V2_1,
                PROMPT_V2_2,
                PROMPT_V2_3,
                PROMPT_V2_4,
            ],
        )


if __name__ == "__main__":
    unittest.main()
