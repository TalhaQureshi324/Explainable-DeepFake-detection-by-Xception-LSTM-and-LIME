import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import cv2

def explain_prediction(model_path, processed_frames):
    """
    processed_frames: The (1, 10, 299, 299, 3) numpy array from test.py
    """
    model = tf.keras.models.load_model(model_path)
    explainer = lime_image.LimeImageExplainer()

    # LIME works on 2D images. Since your model takes 10 frames, 
    # we explain the "most representative" frame (usually the middle one).
    target_frame = processed_frames[0][5] 

    def predict_wrapper(images):
        # LIME provides a batch of images; we must turn them into 
        # the (Batch, 10, 299, 299, 3) format the model expects.
        # We duplicate the single image 10 times to simulate a sequence.
        responses = []
        for img in images:
            seq = np.stack([img] * 10, axis=0) 
            res = model.predict(np.expand_dims(seq, axis=0), verbose=0)
            # Return probabilities for both classes [Real, Fake]
            responses.append([1-res[0][0], res[0][0]])
        return np.array(responses)

    explanation = explainer.explain_instance(
        target_frame.astype('double'), 
        predict_wrapper, 
        top_labels=1, 
        hide_color=0, 
        num_samples=100 # Increase for better quality, decrease for speed
    )

    # Get the heatmap mask
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )

    plt.imshow(mark_boundaries(temp, mask))
    plt.title("XAI Explanation: Highlighted areas indicate Fake features")
    plt.savefig("explanation_result.png")
    print("Explanation saved as explanation_result.png")

# To use this, call it inside your test.py after preprocessing.