import torch

# Load the file
try:
    state_dict = torch.load("model.pt", map_location="cpu")
    print(f"Loaded model.pt successfully.")
    
    # Check for specific keys
    keys = state_dict.keys()
    bank_keys = [k for k in keys if "bank_" in k]
    
    if not bank_keys:
        print("\n❌ CRITICAL ISSUE: No 'bank_feats' or 'bank_coords' found!")
        print("This file contains only weights. You need to re-run the 'Baking' cell in Colab.")
    else:
        print(f"\n✅ Found bank data keys: {bank_keys}")
        print("If these keys have 'module.' in front of them, we need to update model.py.")
        
    # Check shape if exists
    for k in bank_keys:
        print(f"{k}: {state_dict[k].shape}")

except Exception as e:
    print(f"Error loading file: {e}")