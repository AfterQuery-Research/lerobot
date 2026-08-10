from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path


def _input_tree(root: Path, label: str) -> None:
    for directory in ("data", "meta", "videos"):
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{label}.bin").write_bytes(f"{label}-{directory}".encode())


def test_atomic_builder_publishes_read_only_sibling_and_postvalidates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = tmp_path / "source"
    v2_root = tmp_path / "v2"
    final_root = tmp_path / "dual-lidar-umi-currentrel-r6d-onset-v3"
    report_path = tmp_path / "validation.json"
    validation_count_path = tmp_path / "validation-count.txt"
    fake_python = tmp_path / "fake-python"
    _input_tree(source_root, "source")
    _input_tree(v2_root, "v2")
    fake_python.write_text(
        f"""#!{sys.executable}
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path


def value(prefix: str) -> Path:
    return Path(next(arg.split('=', 1)[1] for arg in sys.argv[1:] if arg.startswith(prefix)))


if sys.argv[1:3] == ['-m', 'lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d_onset_v3']:
    output = value('--output-root=')
    for directory in ('data', 'meta', 'videos'):
        path = output / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / 'payload.bin').write_bytes(directory.encode())
elif sys.argv[1].endswith('validate_currentrel_onset_v3_artifact.py'):
    artifact = value('--v3-root=')
    report = value('--report=')
    count_path = Path(os.environ['FAKE_VALIDATION_COUNT'])
    count = int(count_path.read_text()) + 1 if count_path.exists() else 1
    count_path.write_text(str(count))
    paths = [artifact, *artifact.rglob('*')]
    writable = [str(path) for path in paths if stat.S_IMODE(path.stat().st_mode) & 0o222]
    if count == 1 and not writable:
        raise SystemExit('prepublish artifact unexpectedly read-only')
    if count == 2 and writable:
        raise SystemExit(f'postpublish artifact has writable paths: {{writable}}')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({{'passed': True, 'validation_call': count}}))
else:
    raise SystemExit(f'unexpected fake-python arguments: {{sys.argv[1:]}}')
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = {
        **os.environ,
        "UMI_YAM_SOURCE_ROOT": str(source_root),
        "UMI_YAM_ONSET_V2_ROOT": str(v2_root),
        "UMI_YAM_ONSET_V3_ROOT": str(final_root),
        "UMI_YAM_ONSET_V3_REPORT": str(report_path),
        "UMI_YAM_PYTHON": str(fake_python),
        "FAKE_VALIDATION_COUNT": str(validation_count_path),
    }
    result = subprocess.run(
        [str(repo_root / "examples/umi_yam/build_currentrel_onset_v3.sh")],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert final_root.is_dir()
    assert validation_count_path.read_text() == "2"
    assert json.loads(report_path.read_text()) == {"passed": True, "validation_call": 2}
    assert "Validated immutable onset-v3 artifact" in result.stdout
    for path in (final_root, *final_root.rglob("*")):
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    assert not list(tmp_path.glob(f".{final_root.name}.build.*"))
