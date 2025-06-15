import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from huggingface_hub import hf_hub_download

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
