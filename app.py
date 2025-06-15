import os
import io
import torch
import torch.nn as nn
from torchvision import models as torchvision_models # Explicitly use torchvision models
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download # To download the specific .pth file
import os
import json
from groq import Groq
import logging

from google import genai
from google.genai import types

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
API_KEY_1 = os.getenv("API_KEY")
API_KEY_2 = os.getenv("GEMINI")
TASK = os.getenv("TASK")
client = genai.Client(api_key=API_KEY_2)
SYSTEM = os.getenv("SYSTEM")
LANGUAGE_DETECTION_PROMPT = os.getenv("Language")

# ---------- Utility Functions ----------

def extract_json_from_response(response):
    message_body = response.text
    if "```" in message_body:
        start = message_body.find("```")
        end = message_body.find("```", start + 3)
        if end != -1:
            code_block = message_body[start + 3:end].strip()
            if code_block.startswith("json"):
                code_block = code_block[4:].strip()
            message_body = code_block
        else:
            message_body = message_body[start + 3:].strip()

    try:
        parsed_json = json.loads(message_body)
        logging.info("Parsed JSON: %s", parsed_json)
        return parsed_json
    except json.JSONDecodeError as e:
        logging.error("Failed to parse JSON: %s", e)
        return {}

def detect_language(text):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=LANGUAGE_DETECTION_PROMPT),
            contents=[{"role": "user", "parts": [{"text": text}]}]
        )
        detected_json = extract_json_from_response(response)
        return detected_json.get("input_lang", "English")
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "English"

def translate(content, target_language):
    try:
        if not content:
            return {"translation": ""}
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=f"Translate the following content into {target_language}. Respond with plain text only."),
            contents=[{"role": "user", "parts": [{"text": content}]}]
        )
        return extract_json_from_response(response) or {"translation": response.text.strip()}
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return {"translation": content}

# ---------- Message Generator ----------

def gen_message(data_dict, language="English"):
    if not data_dict.get("symptoms") or not data_dict.get("diseases"):
        template = f"""
Hi there, I'm Med-AI-Care, your virtual health assistant.

I understood your message as:  
"{data_dict.get('translation')}"

Thank you for reaching out. I couldn't detect specific symptoms or conditions. It's okay to feel unsure — I'm still here to support you.

My Analysis:  
{data_dict.get('analysis')}

Recommendation:  
{data_dict.get('recommendation')}

Please consult a licensed medical professional for any health concerns.

Take care!
"""
    else:
        symptoms = ', '.join(data_dict.get("symptoms")).capitalize()
        diseases = '\n'.join([f"- {disease}: {likelihood}" for disease, likelihood in data_dict.get("diseases").items()])
        template = f"""
Hi there, I'm Med-AI-Care, your virtual health assistant.

You said:  
"{data_dict.get('translation')}"

Symptoms detected:  
{symptoms}

Possible conditions:  
{diseases}

My Analysis:  
{data_dict.get('analysis')}

Recommendation:  
{data_dict.get('recommendation')}

I'm not a doctor — please follow up with a healthcare professional.

Thanks for reaching out!
"""
    if language != "English":
        translated = translate(template, language)
        return translated.get("translation", template)
    return template

# ---------- Core Chat Function ----------

def chat(cmd):
    language = detect_language(cmd)

    client_instance = Groq(api_key=API_KEY_1)
    completion = client_instance.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": TASK},
            {"role": "user", "content": cmd}
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )

    output = []
    for chunk in completion:
        output.append(chunk.choices[0].delta.content or "")

    # Extract JSON from response
    result = ""
    in_block = False
    for line in output:
        if line.strip() == "```":
            in_block = not in_block
            continue
        if in_block and line.strip().lower() != "json":
            result += line

    try:
        result_json = json.loads(result)
        data_dict = result_json if language == "English" else translate(result, language)
        message_body = gen_message(data_dict, language)
    except json.JSONDecodeError as e:
        logging.error("Final JSON parsing failed: %s", e)
        data_dict = {}
        message_body = "Sorry, I had trouble understanding your input."

    return output, data_dict, message_body

# --- Model Definition (Must match the architecture used for training) ---
def load_model_from_hf(repo_id, filename, num_classes, device):
    model_weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Downloaded model weights to: {model_weights_path}")

    model = torchvision_models.convnext_tiny(weights=None)

    original_classifier_structure = model.classifier
    num_features = original_classifier_structure[2].in_features

    model.classifier = nn.Sequential(
        original_classifier_structure[0],
        original_classifier_structure[1],
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(num_features, num_classes)
    )

    try:
        state_dict = torch.load(model_weights_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        raise

    model.to(device)
    model.eval()
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

# Optional: Set up logging if required for production.
logging.basicConfig(level=logging.INFO)

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

@app.route("/chat",methods=["POST"])
def get_response():
    data = request.get_json()
    query = data.get("query")
    logging.info(f"User Input : {query}")

    output,data_dict,message_body = chat(query)
    return jsonify({"Output" : message_body})

@app.route("/", methods=["GET"])
def home():
    return "ConvNeXt Skin Disease Prediction API. Use POST to /predict with an image file."

