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

# --- Model Definition (Must match the architecture used for training) ---
from model_load import load_model_from_hf

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

def extract_json_from_response(response):
    message_body = response.text

    # Try to locate code block with triple backticks
    if "```" in message_body:
        start = message_body.find("```")
        end = message_body.find("```", start + 3)
        
        if end != -1:
            # Strip optional 'json' or other format specifier
            code_block = message_body[start + 3:end].strip()
            if code_block.startswith("json"):
                code_block = code_block[4:].strip()
            message_body = code_block
        else:
            # Only one set of backticks found, possibly malformed
            message_body = message_body[start + 3:].strip()
    
    # If there's no backtick block, assume it's plain JSON
    try:
        parsed_json = json.loads(message_body)
        logging.info("Parsed JSON: %s", parsed_json)
        return parsed_json
    except json.JSONDecodeError as e:
        logging.error("Failed to parse JSON: %s", e)
        return None
    

def translate(message):
    try:
        # Request to generate content using the model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=SYSTEM),
            contents=message
        )
        

        message_body = extract_json_from_response(response)

        return message_body
    
    except Exception as e:
        logging.error(f"Error while translating message: {e}")
        return {}  # Return an empty dict in case of any unexpected errors

def gen_message(data_dict):
    logging.info(data_dict)
    if not data_dict['symptoms'] or not data_dict['diseases']:
        message_body = f"""
    Hi there, I'm Med-AI-Care, your virtual health assistant.

I understood your message as:  
**"{data_dict['translation']}"**

Thank you for reaching out. Right now, I couldn't detect specific symptoms or conditions based on your message. But you're not alone — it's okay to feel unsure, and I'm still here to support you.

**My Analysis**:  
{data_dict['analysis']}

**Recommendation**:  
{data_dict['recommendation']}

I'm an AI assistant, not a doctor. For any health concern, please consult a licensed medical professional.

Take care, and feel free to share more if you'd like me to help further.
    """.replace("*","\n")
    else:
        message_body = f"""
    Hi there, I'm Med-AI-Care, your virtual health assistant.

You said:  
**"{data_dict['translation']}"**

Here are the symptoms I picked up:  
**{', '.join(data_dict['symptoms']).capitalize()}**

Possible conditions based on your symptoms:  
{chr(10).join([f"- {disease}: {likelihood}" for disease, likelihood in data_dict['diseases'].items()])}

**My Analysis**:  
{data_dict['analysis']}

📝 **Recommendation**:  
{data_dict['recommendation']}

I'm just an AI helper — not a doctor. Please follow up with a healthcare professional for proper diagnosis and treatment.

Thanks for reaching out. I'm here if you need more help.
    """.replace("*","\n")
        
        return message_body

def chat(cmd):
    client = Groq(api_key=API_KEY_1)
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
        {
            "role": "system",
            "content": TASK
        },
        {
            "role": "user",
            "content": cmd
        }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    output = []
    for chunk in completion:
        output.append(chunk.choices[0].delta.content or "")


    in_block = False
    result = ""

    for line in output:
        if line.strip() == "```":
            if not in_block:
                in_block = True
            else:
                break
        elif in_block and line.strip().lower() != "json":
            result+=line


    try:
        result_json = json.loads(result)
        print(result_json)
        lang = result_json["input_lang"]
        if lang != "English":
            data_dict = translate(result)
            message_body = gen_message(data_dict)
        else:
            data_dict = result_json
            message_body = gen_message(data_dict)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        output,data_dict,message_body = chat(cmd)
    
    return output,data_dict,message_body



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

    output,data_dict,message_body = chat(query)
    return jsonify({"Output" : message_body})

@app.route("/", methods=["GET"])
def home():
    return "ConvNeXt Skin Disease Prediction API. Use POST to /predict with an image file."

