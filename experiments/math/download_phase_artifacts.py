#!/usr/bin/env python3
"""Package SDPO-Math run artifacts into a downloadable tar.gz archive."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINED_VARIANTS = {"base_rl", "sdpo_vanilla", "sdpo_reliability", "sdpo_reliability_gate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, help="Phase log directory. Defaults to latest thesis run.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "downloads")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--require-checkpoints", action="store_true")
    return parser.parse_args()


def default_log_dir() -> Path:
    latest_thesis = PROJECT_ROOT / "logs/sdpo_math_phase/latest_thesis_log_dir.txt"
    if latest_thesis.exists():
        return Path(latest_thesis.read_text(encoding="utf-8").strip())

    candidates = [path for path in (PROJECT_ROOT / "logs/sdpo_math_phase").glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError("no SDPO-Math phase log directories found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_manifest(log_dir: Path) -> dict[str, Any]:
    manifest_path = log_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def latest_checkpoint_dir(root: Path) -> Path | None:
    tracker = root / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        step = tracker.read_text(encoding="utf-8").strip()
        candidate = root / f"global_step_{step}"
        if candidate.exists():
            return candidate

    candidates = [path for path in root.glob("global_step_*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_step)


def add_path(archive: tarfile.TarFile, path: Path, archive_root: str) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = Path(resolved.name)
    archive.add(resolved, arcname=str(Path(archive_root) / relative))


def main() -> None:
    args = parse_args()
    log_dir = (args.log_dir or default_log_dir()).resolve()
    manifest = load_manifest(log_dir)
    exp_suffix = str(manifest["exp_suffix"])
    phase = str(manifest.get("phase", "phase"))
    seed = str(manifest.get("seed", "seed"))
    model = slug(str(manifest.get("model", "model")).replace("/", "_"))
    created = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_root = f"sdpo_math_{slug(phase)}_{model}_seed{seed}_{created}"
    archive_path = args.output_dir / f"{archive_root}.tar.gz"

    checkpoint_roots: list[Path] = []
    if args.include_checkpoints or args.require_checkpoints:
        project_root = Path(manifest.get("project_root") or PROJECT_ROOT)
        for variant in manifest.get("variants", []):
            if variant not in TRAINED_VARIANTS:
                continue
            checkpoint_root = project_root / "checkpoints/sdpo_math" / f"{variant}_{exp_suffix}"
            latest = latest_checkpoint_dir(checkpoint_root)
            if latest is None:
                if args.require_checkpoints:
                    raise FileNotFoundError(f"missing checkpoint for {variant}: {checkpoint_root}")
                continue
            checkpoint_roots.append(latest)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_dir": str(log_dir),
        "archive": str(archive_path),
        "included_checkpoints": [str(path) for path in checkpoint_roots],
    }
    metadata_path = args.output_dir / f"{archive_root}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with tarfile.open(archive_path, "w:gz") as archive:
        for path in [
            log_dir / "manifest.json",
            log_dir / "summary.csv",
            log_dir / "summary.md",
            metadata_path,
        ]:
            add_path(archive, path, archive_root)
        for subdir in ["metrics", "validation"]:
            add_path(archive, log_dir / subdir, archive_root)
        for path in sorted(log_dir.glob("*.log")):
            add_path(archive, path, archive_root)
        for checkpoint_root in checkpoint_roots:
            add_path(archive, checkpoint_root, archive_root)
            tracker = checkpoint_root.parent / "latest_checkpointed_iteration.txt"
            add_path(archive, tracker, archive_root)

    print(f"archive={archive_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
