#!/usr/bin/env python
"""Read-only onset-v3 training input gate expected by the smoke/full sbatches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_currentrel_onset_v3_artifact import validate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--v2-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve(strict=True)
    manifest = json.loads((dataset_root / "meta/artifact_manifest.json").read_text(encoding="utf-8"))
    source_root = args.source_root or Path(manifest["source"]["root"])
    v2_root = args.v2_root or dataset_root.with_name("dual-lidar-umi-currentrel-r6d-onset-v2")
    report = validate(source_root, v2_root, dataset_root)
    compact = {
        "passed": report["passed"],
        "v3_checksum_list_sha256": report["v3_checksum_list_sha256"],
        "valid_action_positions": report["valid_action_positions"],
        "optimizer_contact_anchors_preserved": report["optimizer_contact_anchors_preserved"],
        "detected_gripper_event_frames_preserved": report["detected_gripper_event_frames_preserved"],
        "video_files_sha256_equal": report["video_files_sha256_equal"],
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
