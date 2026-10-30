"""Save compact, versioned benchmark evidence after a completed experiment."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int)
    parser.add_argument("--disposition", choices=["validated", "experimental"], default="validated")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    cases = []
    for name in ["decode", "small", "prefill-k1024", "prefill-k2048"]:
        data = json.loads((args.evidence / f"{name}.json").read_text(encoding="utf-8"))
        assert len(data["event_samples_us"]) == 50
        assert len(data["kernel_samples_us"]) >= 10
        cases.append(data)
    assert len({case["kernel_source_sha256"] for case in cases}) == 1
    assert cases[2]["input_sha256"] == cases[3]["input_sha256"]
    if args.tests_passed is not None:
        log = (args.evidence / "tests.log").read_text(encoding="utf-8")
        assert f"{args.tests_passed} passed" in log
    relative = "src/tileops/_kernels/attention/topk_select_ascend.py"
    if args.version == "v0.1.0":
        source = subprocess.run(
            ["git", "show", f"topk-npu-v0.1.0:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    else:
        source = (repo / relative).read_bytes()
        assert hashlib.sha256(source).hexdigest() == cases[0]["kernel_source_sha256"]
    normalized = source.replace(b"\r\n", b"\n")
    result = {
        "tag": "topk-npu-" + args.version,
        "disposition": args.disposition,
        "kernel_path": relative,
        "kernel_normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "kernel_git_blob_sha1": hashlib.sha1(
            b"blob " + str(len(normalized)).encode() + b"\0" + normalized
        ).hexdigest(),
        "npu_tests_passed": args.tests_passed,
        "cases": cases,
    }
    directory = repo / "docs" / "performance" / "topk-select-950pr"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{args.version}.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
