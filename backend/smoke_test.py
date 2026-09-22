"""Phase 1 gate. Exercises inference.py directly -- no server involved.

    python smoke_test.py test.jpg

Passing this is the precondition for touching main.py. Debugging segmentation
through HTTP is much slower than debugging it here.
"""

from __future__ import annotations

import json
import sys

import inference


def main(path: str) -> int:
    print(f"weights : {inference.model_info()['weights']}")
    print(f"image   : {path}\n")

    with open(path, "rb") as fh:
        data = fh.read()

    result = inference.analyse_bytes(data, conf=0.25, include_debug=True)

    if not result["ok"]:
        print(f"FAIL - {result['error']}")
        return 1

    checks = {
        "ok is True": result["ok"] is True,
        "latency under 1000 ms": result["latency_ms"] < 1000,
        "annotated_png present": bool(result.get("annotated_png")),
        "debug panels present": len(result.get("debug", {})) == 3,
        "lesion fields complete": all(
            {"max_diameter_px", "area_px", "circularity",
             "solidity", "echo_mean", "echo_sd"} <= set(les)
            for les in result["lesions"]
        ),
    }

    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\ndetections : {result['count']}")
    print(f"latency    : {result['latency_ms']} ms")
    if result["lesions"]:
        print("\nfirst lesion:")
        print(json.dumps(result["lesions"][0], indent=2))

    # Write the annotated frame out so the mask can be inspected by eye.
    if result.get("annotated_png"):
        import base64
        with open("smoke_out.png", "wb") as fh:
            fh.write(base64.b64decode(result["annotated_png"]))
        print("\nannotated image written to smoke_out.png")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
