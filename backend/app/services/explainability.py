"""
Explainability service (LIME).

Optional and failure-isolated: an XAI error must never invalidate the base
prediction. The current implementation explains a SINGLE replicated frame of
the sequence — it attributes the spatial pathway's response only, NOT the
temporal (LSTM) pathway; every artifact carries that limitation text.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("deepfake.explainability")

XAI_METHOD = "LIME"
XAI_LIMITATION = (
    "LIME highlights regions of a single replicated frame that most influenced "
    "the model's response. It does NOT prove those pixels were manipulated and "
    "does not explain the temporal (LSTM) pathway."
)


def lime_available() -> bool:
    try:
        import lime  # noqa: F401
        from skimage.segmentation import mark_boundaries  # noqa: F401
        return True
    except ImportError:
        return False


def generate_lime_heatmap(model, num_frames: int, frames_rgb, work_dir: Path,
                          num_samples: int | None = None) -> Optional[dict]:
    """Return {"artifact_id", "png_path", "image_size"} or raise on failure."""
    if not lime_available():
        raise RuntimeError("lime / scikit-image are not installed")

    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    from PIL import Image

    if num_samples is None:
        num_samples = int(os.environ.get("DEEPFAKE_LIME_SAMPLES", "80"))

    normalized = [f.astype(np.float32) / 255.0 for f in frames_rgb]
    padded = normalized[:]
    while len(padded) < num_frames:
        padded.append(padded[-1].copy())
    target_frame = padded[num_frames // 2]

    explainer = lime_image.LimeImageExplainer()

    def predict_wrapper(images):
        responses = []
        for img in images:
            seq = np.stack([img] * num_frames, axis=0)
            res = model.predict(np.expand_dims(seq, axis=0), verbose=0)
            responses.append([1 - res[0][0], res[0][0]])
        return np.array(responses)

    explanation = explainer.explain_instance(
        target_frame.astype("double"), predict_wrapper,
        top_labels=1, hide_color=0, num_samples=num_samples,
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False,
    )
    out_img = (mark_boundaries(temp, mask) * 255).astype(np.uint8)

    xai_dir = Path(work_dir) / "artifacts"
    xai_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = "xai_" + os.urandom(6).hex()
    png_path = xai_dir / f"{artifact_id}.png"
    Image.fromarray(out_img).save(png_path)
    return {"artifact_id": artifact_id, "png_path": str(png_path), "image_size": out_img.shape[:2]}
