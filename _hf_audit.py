"""Phase A audit: live smoke tests against the deployed Hugging Face Space."""
import json
import re
import time
from pathlib import Path

from gradio_client import Client, handle_file

SPACE = "MuhammadTalha121/Deep-Fake-Detection"
MEDIA = Path("e2e_test_media")

out = {"space": SPACE, "tests": []}
t0 = time.time()
print("Connecting (includes cold start if the Space was sleeping)...")
client = Client(SPACE)
out["connect_seconds"] = round(time.time() - t0, 1)
print(f"connected in {out['connect_seconds']}s")


def run(label, file_path, enable_xai, timeout_s=600):
    t = time.time()
    try:
        result = client.predict(
            input_file=handle_file(str(file_path)),
            enable_xai=enable_xai,
            lang_display="English",
            api_name="/predict_deepfake",
        )
    except Exception as exc:  # noqa: BLE001 - record failures as evidence
        out["tests"].append({"label": label, "error": str(exc)[:400],
                             "seconds": round(time.time() - t, 1)})
        print(f"[{label}] ERROR after {time.time()-t:.1f}s: {str(exc)[:200]}")
        return None

    md, face, lime = result
    seconds = round(time.time() - t, 1)
    entry = {
        "label": label,
        "xai": enable_xai,
        "seconds": seconds,
        "markdown": md,
        "face_file": str(face),
        "lime_file": str(lime) if lime else None,
    }
    out["tests"].append(entry)
    print(f"[{label}] {seconds}s")
    print("  markdown:", re.sub(r"\s+", " ", md)[:220])
    print("  face:", str(face)[:100])
    print("  lime:", (str(lime)[:100] if lime else "None"))
    return entry


# 1) image without LIME
run("image_no_xai", MEDIA / "test_face.jpg", False)
# 2) video without LIME
run("video_no_xai", MEDIA / "test_video.mp4", False)
# 3) image WITH LIME (slow path)
run("image_with_xai", MEDIA / "test_face.jpg", True)

Path("e2e_test_media/hf_audit.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("\naudit saved -> e2e_test_media/hf_audit.json")
