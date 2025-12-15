import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from typing import Tuple, List

def prepare_data(path: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Reads a CSV file or directory, loads images, applies inference transforms.
    Returns X (list of tensors) and y (list of [lat, lon]).
    """
    
    # 1. Define Inference Transform (Must match your validation transform)
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    X = []
    y = []

    # Check if path is a file (CSV) or directory
    if os.path.isfile(path):
        # Assume CSV format: "image_path, latitude, longitude"
        df = pd.read_csv(path)
        
        # Determine root directory (usually usually the same folder as the CSV)
        root_dir = os.path.dirname(path)
        
        for _, row in df.iterrows():
            # Handle image path (sometimes absolute, sometimes relative)
            img_name = row.iloc[0] # Assuming first column is filename
            
            # Construct full path
            full_img_path = os.path.join(root_dir, img_name)
            
            # Handle case where the CSV might contain full paths already
            if not os.path.exists(full_img_path):
                if os.path.exists(img_name):
                    full_img_path = img_name
            
            try:
                image = Image.open(full_img_path).convert("RGB")
                image = inference_transform(image)
                
                # GPS coords
                lat = float(row.iloc[1])
                lon = float(row.iloc[2])
                
                X.append(image)
                y.append(torch.tensor([lat, lon]))
            except Exception as e:
                print(f"Error loading {full_img_path}: {e}")
                
    else:
        # Fallback: if they point to a folder, usually we just list images
        # But usually in these comps, path is a CSV.
        raise ValueError("prepare_data expected a CSV file path.")

    return X, y