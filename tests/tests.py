import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras


CLASS_NAMES = ["InventoryMovements", "Invoices", "PurchaseOrders", "ShippingOrders"]
INVOICE_LABELS = {"invoice", "invoices"}
IMG_HEIGHT, IMG_WIDTH = 224, 224


def extract_and_save_invoice_data(image_path):
    """Handle invoice extraction when an extraction implementation is configured."""
    print(f"[INFO] Invoice extraction is not configured for {os.path.basename(image_path)}.")


# ==========================================
# 4. INFERENCE PIPELINE WITH CONDITION TRIGGER
# ==========================================
def process_new_document(image_path, trained_model):
    """Predict a document class and trigger invoice extraction only for invoices."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    predictions = trained_model.predict(image_array, verbose=0)
    score = predictions[0]
    predicted_class_idx = int(np.argmax(score))
    predicted_label = CLASS_NAMES[predicted_class_idx]
    confidence = float(100 * np.max(score))

    results = {
        "document": os.path.basename(image_path),
        "classification": predicted_label,
        "confidence": confidence,
    }

    if predicted_label in {"Invoice", "Invoices"}:
        extract_and_save_invoice_data(image_path)
        results["action"] = "extracted_invoice_data"
    else:
        results["action"] = "skipped_non_invoice"

    return results


if __name__ == "__main__":
    loaded_model_path = "model.keras"

    if not os.path.exists(loaded_model_path):
        print(f"Model not found at {loaded_model_path}")
        sys.exit(1)

    classifier_model = keras.models.load_model(loaded_model_path)
    print(f"Loaded pre-trained model from: {loaded_model_path}")

    for mock_file in ["test_invoice.png", "test_purchase_order.png"]:
        tf.keras.preprocessing.image.save_img(
            mock_file,
            np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8),
        )
        try:
            analysis = process_new_document(mock_file, classifier_model)
            print("\n--- Analysis Results ---")
            print(f"Document: {analysis['document']}")
            print(f"Classification: {analysis['classification']} ({analysis['confidence']:.2f}% confidence)")
            print(f"Action: {analysis['action']}")
        finally:
            if os.path.exists(mock_file):
                os.remove(mock_file)
