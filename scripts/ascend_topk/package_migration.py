"""Package the current source tree and selected public experiment inputs."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    output = repo / "artifacts" / "topk-migration-20261031"
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    for name in tracked:
        if name.startswith(("src/", "tests/", "workloads/")) or name in {
            "conftest.py", "pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES"
        } or name.startswith("scripts/ascend_topk/"):
            path = repo / name
            if path.is_file():
                files["TileOPs/" + name] = path.read_bytes()
    guide = "docs/design/topk-select-950pr-private-server-guide.md"
    files["TileOPs/" + guide] = (repo / guide).read_bytes()
    for name in ["control_portable.py", "package_migration.py"]:
        files["TileOPs/scripts/ascend_topk/" + name] = (repo / "scripts/ascend_topk" / name).read_bytes()
    kernel = "src/tileops/_kernels/attention/topk_select_ascend.py"
    for version in ["v0.1.0", "v0.1.2"]:
        files[f"support/kernel-{version}.py"] = subprocess.check_output(
            ["git", "show", f"topk-npu-{version}:{kernel}"], cwd=repo
        )
    baseline = repo / "artifacts/cann-topkv2-static-20261030"
    for name in ["build.cpp", "top_k_v2_proto.h", "benchmark.py", "profile-summary.py"]:
        files["support/cann/" + name] = (baseline / name).read_bytes()
    for name in ["measure.py", "record.py"]:
        path = "TileOPs/scripts/ascend_topk/" + name
        source = files[path].decode()
        if name == "measure.py":
            source = source.replace('"physical_device": 1,',
                                    '"physical_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],')
        files[path] = source.encode()
    # Keep the legacy baseline intact except for truthful device metadata.
    path = "support/cann/benchmark.py"
    files[path] = files[path].decode().replace(
        "physical_device=1,", 'physical_device=os.environ["ASCEND_RT_VISIBLE_DEVICES"],'
    ).encode()
    # Shells and Python sources travel with stable LF bytes.
    files = {name: data.replace(b"\r\n", b"\n") if name.endswith((".py", ".sh", ".md"))
             else data for name, data in files.items()}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    manifest["_note"] = "Current worktree candidate; not a validated v0.1.4 release."
    files["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    target = output / "topk-private-server-kit.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    print(target)
    print("sha256", hashlib.sha256(target.read_bytes()).hexdigest())
    print("files", len(files))


if __name__ == "__main__":
    main()
