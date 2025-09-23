# Save as check_model.py and run it
import torch
import os

# Path to your model
model_path = "/Users/sachin/Downloads/CaseAnalytics-main/backend/models/weapon_detect.pt"

# Check if model exists
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    exit(1)

try:
    # Try loading with torch.hub
    print("Attempting to load model with torch.hub...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
    print("Success! Model loaded with torch.hub")
    if hasattr(model, 'names'):
        print(f"Model classes: {model.names}")
    else:
        print("Warning: Model has no 'names' attribute")
    
except Exception as e:
    print(f"Failed to load with torch.hub: {e}")
    
    try:
        # Try direct loading
        print("\nAttempting to load directly with torch.load...")
        model = torch.load(model_path, map_location='cpu')
        print("Success! Model loaded with torch.load")
        print(f"Model type: {type(model)}")
    except Exception as e2:
        print(f"Failed to load with torch.load: {e2}")
        print("\nYour model may be corrupted or incompatible.")