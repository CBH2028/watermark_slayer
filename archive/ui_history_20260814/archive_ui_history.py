from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT / "archive" / "ui_history_20260814"
SNAPSHOT_DIR = ARCHIVE / "pre_cross_platform_snapshot"
LEGACY_DIR = ARCHIVE / "legacy_original"

SNAPSHOT_NAMES = (
    "README.md",
    "download.sh",
    "requirements.txt",
    "setup.sh",
    "setup.bat",
    "setup.ps1",
    "run.sh",
    "run.bat",
    "ui",
    "ui.yml",
    "watermark_slayer_gui.py",
    "watermark_slayer.py",
    "florence_od_runtime.py",
    "florence_od_test.py",
    "assets/clip-pre.mp4",
    "assets/clip-after.mp4",
)

LEGACY_NAMES = (
    "run_original.sh",
    "ui_original",
    "ui_original.yml",
    "watermark_slayer_gui_original.py",
    "watermark_slayer_original.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive UI history before the cross-platform launcher update."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create the snapshot and move legacy UI files.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild metadata for an existing archive.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def collect_file_inventory() -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for root in (SNAPSHOT_DIR, LEGACY_DIR):
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            inventory.append(
                {
                    "path": str(path.relative_to(ARCHIVE)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return inventory


def refresh_metadata() -> None:
    expected = (
        *(("copy", name, SNAPSHOT_DIR / name) for name in SNAPSHOT_NAMES),
        *(("move", name, LEGACY_DIR / name) for name in LEGACY_NAMES),
    )
    missing = [str(destination) for _, _, destination in expected if not destination.exists()]
    if missing:
        raise FileNotFoundError("Archived entries are missing:\n" + "\n".join(missing))

    manifest_path = ARCHIVE / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = collect_file_inventory()
    payload.update(
        {
            "entry_count": len(expected),
            "file_count": len(inventory),
            "total_size_bytes": sum(int(item["size_bytes"]) for item in inventory),
            "entries": [
                {
                    "operation": operation,
                    "source": name,
                    "destination": str(destination.relative_to(PROJECT)),
                }
                for operation, name, destination in expected
            ],
            "files": inventory,
        }
    )
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme_path = ARCHIVE / "README.md"
    readme_lines = readme_path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "Files:": f"Files: `{payload['file_count']}`",
        "Total bytes:": f"Total bytes: `{payload['total_size_bytes']}`",
    }
    updated_lines = [
        next(
            (replacement for prefix, replacement in replacements.items() if line.startswith(prefix)),
            line,
        )
        for line in readme_lines
    ]
    readme_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def validate() -> None:
    missing = [
        str(PROJECT / name)
        for name in (*SNAPSHOT_NAMES, *LEGACY_NAMES)
        if not (PROJECT / name).exists()
    ]
    if missing:
        raise FileNotFoundError("Archive sources are missing:\n" + "\n".join(missing))
    for destination in (SNAPSHOT_DIR, LEGACY_DIR):
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")


def describe() -> None:
    print(f"archive: {ARCHIVE}")
    print(f"snapshot entries: {len(SNAPSHOT_NAMES)}")
    for name in SNAPSHOT_NAMES:
        print(f"[copy] {PROJECT / name} -> {SNAPSHOT_DIR / name}")
    print(f"legacy entries: {len(LEGACY_NAMES)}")
    for name in LEGACY_NAMES:
        print(f"[move] {PROJECT / name} -> {LEGACY_DIR / name}")


def execute() -> None:
    copied: list[Path] = []
    moved: list[tuple[Path, Path]] = []
    records: list[dict[str, str]] = []
    try:
        for name in SNAPSHOT_NAMES:
            source = PROJECT / name
            destination = SNAPSHOT_DIR / name
            copy_path(source, destination)
            copied.append(destination)
            records.append(
                {
                    "operation": "copy",
                    "source": name,
                    "destination": str(destination.relative_to(PROJECT)),
                }
            )

        for name in LEGACY_NAMES:
            source = PROJECT / name
            destination = LEGACY_DIR / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
            records.append(
                {
                    "operation": "move",
                    "source": name,
                    "destination": str(destination.relative_to(PROJECT)),
                }
            )
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        for destination in reversed(copied):
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
        raise

    inventory = collect_file_inventory()
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "archive_name": ARCHIVE.name,
        "created_at": created_at,
        "policy": (
            "The active UI was copied as a pre-cross-platform snapshot. Historical "
            "original UI files were moved without deletion."
        ),
        "entry_count": len(records),
        "file_count": len(inventory),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in inventory),
        "entries": records,
        "files": inventory,
    }
    with (ARCHIVE / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# UI History Archive",
        "",
        "本目录保存跨平台改造前的界面版本。所有内容均有 SHA256 记录，未删除历史文件。",
        "",
        "## Contents",
        "",
        "- `pre_cross_platform_snapshot`: 当前正式 UI 在跨平台改造前的完整快照。",
        "- `legacy_original`: 早期双类别原版 UI、对应处理程序和启动脚本。",
        "- `manifest.json`: 原路径、归档路径、文件大小和 SHA256。",
        "",
        "## Restore",
        "",
        "需要恢复时，按照 `manifest.json` 中的记录操作。`move` 项可反向移动，"
        "`copy` 项可覆盖回原路径。",
        "",
        f"Archived at: `{created_at}`",
        f"Files: `{len(inventory)}`",
        f"Total bytes: `{payload['total_size_bytes']}`",
        "",
    ]
    (ARCHIVE / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if args.refresh:
        refresh_metadata()
        print("UI history manifest refreshed")
        return
    validate()
    describe()
    if not args.execute:
        print("dry-run only; pass --execute to archive these entries")
        return
    execute()
    print("UI history archive completed")


if __name__ == "__main__":
    main()
