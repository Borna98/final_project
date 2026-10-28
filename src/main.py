import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. SIMULATED DATA & CONFIGURATION
# ==========================================
# Map your class folders to integers alphabeticially: 
# 0: InventoryMovements, 1: Invoices, 2: PurchaseOrders, 3: ShippingOrders

CLASS_NAMES = ['InventoryMovements', 'Invoices', 'PurchaseOrders', 'ShippingOrders']
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# ==========================================
# 2. MODEL ARCHITECTURE (Transfer Learning)
# ==========================================
def build_document_classifier():
    # Use MobileNetV2 for an efficient, lightweight document classifier
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base weights Initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation='softmax') # 4 outputs
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# 3. TRIGGER ACTION: DATA EXTRACTION & SAVING
# ==========================================
def extract_and_save_invoice_data(image_path):
    """
    Triggered only if the image is classified as an Invoice.
    In production, swap mock data with OCR libraries like Tesseract or EasyOCR.
    """
    print(f"\n[TRIGGER] Processing Invoice data extraction for: {image_path}")
    
    # Mock extracted database entry payload
    extracted_data = {
        "document_type": "Invoice",
        "file_source": os.path.basename(image_path),
        "invoice_number": "INV-2026-99412",  # Placeholder OCR extraction
        "extracted_amount": 1250.00,
        "status": "Pending_Review"
    }
    
    # Simulate saving payload to a database or JSON file
    output_log = f"invoice_extracted_{os.path.getctime(image_path)}.json"
    print(f"[SUCCESS] Saving invoice records directly to database format...")
    print(f"Payload: {extracted_data}")
    
    return extracted_data


# ==========================================
# 5. EXECUTION BOOTSTRAP
# ==========================================
if __name__ == "__main__":
    
    # Initialize the model blueprint
    classifier_model = build_document_classifier()
    print("Model initialized successfully.")
    
    # Note: To train this model on your local document folders:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'data/', 
        validation_split=0.2, 
        subset="training", 
        seed=123, 
        image_size=(IMG_HEIGHT, IMG_WIDTH), 
        batch_size=BATCH_SIZE
    )
    
    classifier = classifier_model.fit(train_ds, epochs=10)
    
    # Save the trained model for inference 
    classifier_model.save('models/document_classifier_model.keras')
    
    classifier.his
    
    
    