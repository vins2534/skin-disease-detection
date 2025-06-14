import os
import io
import torch
import torch.nn as nn
from torchvision import models as torchvision_models # Explicitly use torchvision models
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download # To download the specific .pth file

# --- Configuration ---
# HF_MODEL_ID = "vinit2534/convnext-skin-disease"
# MODEL_FILENAME_ON_HUB = "best_convnext_10class_skin_model.pth" # The actual filename on the Hub

CLASS_NAMES = [ # Ensure this matches the model you trained and uploaded
    "Acne and Rosacea Photos", "Eczema Photos",
    "Light Diseases and Disorders of Pigmentation", "Melanoma Skin Cancer Nevi and Moles",
    "Psoriasis pictures Lichen Planus and related diseases", "Seborrheic Keratoses and other Benign Tumors",
    "Tinea Ringworm Candidiasis and other Fungal Infections", "Urticaria Hives",
    "Vascular Tumors", "Warts Molluscum and other Viral Infections"
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Definition (Must match the architecture used for training) ---
def load_model_from_hf(repo_id, filename, num_classes, device):
    # Download the model weights file from Hugging Face Hub
    model_weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Downloaded model weights to: {model_weights_path}")

    # Instantiate the ConvNeXt-Tiny architecture
    model = torchvision_models.convnext_tiny(weights=None) # Load architecture
    
    # Rebuild the classifier head exactly as it was during training for the 10 classes
    original_classifier_structure = model.classifier # This is nn.Sequential(LayerNorm2d, Flatten, Linear)
    num_features = original_classifier_structure[2].in_features # Get in_features from the original Linear layer

    model.classifier = nn.Sequential(
        original_classifier_structure[0], # Keep original LayerNorm2d
        original_classifier_structure[1], # Keep original Flatten
        nn.Dropout(p=0.4, inplace=False), # The dropout you added during training
        nn.Linear(num_features, num_classes) # New Linear layer for your 10 classes
    )
    
    # Load the downloaded state dictionary
    try:
        state_dict = torch.load(model_weights_path, map_location=device)
        # If the saved file is a checkpoint dictionary, extract model_state_dict
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            print(f"Loaded model_state_dict from checkpoint: {filename}")
        else: # Assuming it's a raw state_dict
            model.load_state_dict(state_dict)
            print(f"Loaded raw state_dict from: {filename}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture defined here matches the saved model's architecture.")
        print("And that the .pth file on the Hub is a compatible state_dict.")
        raise
        
    model.to(device)
    model.eval() # Set to evaluation mode
    return model

# --- Image Transformations (Must match validation/test transforms used during training) ---
def transform_image(image_bytes):
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return val_test_transform(image).unsqueeze(0) # Add batch dimension

# --- Load Model Globally ---
# Define these based on your Hugging Face repo
HF_REPO_ID = "vinit2534/convnext-skin-disease"
MODEL_FILENAME = "best_convnext_10class_skin_model.pth" # MAKE SURE THIS IS THE EXACT FILENAME on the Hub

try:
    print(f"Attempting to load model from Hugging Face Hub: {HF_REPO_ID}/{MODEL_FILENAME}")
    model_instance = load_model_from_hf(HF_REPO_ID, MODEL_FILENAME, NUM_CLASSES, DEVICE)
    print(f"Model loaded from Hugging Face Hub and ready on {DEVICE}.")
except Exception as e:
    print(f"Failed to load the model on startup: {e}")
    model_instance = None

# --- Flask App Initialization ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if model_instance is None: # Use the globally loaded model
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes).to(DEVICE)
            
            with torch.no_grad():
                outputs = model_instance(tensor) # Use the globally loaded model
                probabilities = torch.softmax(outputs, dim=1)[0]
                _, predicted_idx = torch.max(probabilities, 0)
                predicted_class = CLASS_NAMES[predicted_idx.item()]
                confidence = probabilities[predicted_idx.item()].item()

            class_probabilities = {CLASS_NAMES[i]: probabilities[i].item() for i in range(NUM_CLASSES)}
            
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities
            })
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    return jsonify({"error": "Invalid request"}), 400


@app.route("/", methods=["GET"])
def home():
    return "ConvNeXt Skin Disease Prediction API. Use POST to /predict with an image file."

if __name__ == "__main__":
    # Make sure you have HUGGING_FACE_HUB_TOKEN set as an environment variable if your repo is private
    # For public repos, it's not strictly necessary but good practice for higher rate limits.
    # Example: os.environ['HUGGING_FACE_HUB_TOKEN'] = "your_hf_token_here"
    
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000)