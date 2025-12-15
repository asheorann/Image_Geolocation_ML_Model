import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import Any, Iterable, List
import numpy as np

# --- 1. The Backbone Class (Unchanged) ---
class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None) 
        
        n_inputs = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        # Extract features for k-NN
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x) # LayerNorm
        x = self.backbone.classifier[1](x) # Flatten
        return x

# --- 2. The Wrapper Class ---
class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = CustomConvNeXt(num_classes=2)
        
        # --- FIXED BUFFERS ---
        # Keeping these exact shapes so the evaluator doesn't crash
        self.register_buffer("bank_feats", torch.zeros(740, 768))
        self.register_buffer("bank_coords", torch.zeros(740, 2))
        
        self.register_buffer("lat_mean", torch.tensor(0.0))
        self.register_buffer("lat_std", torch.tensor(1.0))
        self.register_buffer("lon_mean", torch.tensor(0.0))
        self.register_buffer("lon_std", torch.tensor(1.0))

    def eval(self) -> None:
        self.model.eval()

    def load_state_dict(self, state_dict, strict=True):
        if "bank_feats" in state_dict:
            self.bank_feats = state_dict["bank_feats"]
        if "bank_coords" in state_dict:
            self.bank_coords = state_dict["bank_coords"]
            
        if "lat_mean" in state_dict: self.lat_mean = state_dict["lat_mean"]
        if "lat_std" in state_dict: self.lat_std = state_dict["lat_std"]
        if "lon_mean" in state_dict: self.lon_mean = state_dict["lon_mean"]
        if "lon_std" in state_dict: self.lon_std = state_dict["lon_std"]

        return self.model.load_state_dict(state_dict, strict=False)

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        device = next(self.model.parameters()).device
        
        # 1. Handle Input List (Fix for conv2d error)
        if isinstance(batch, list):
            input_tensor = torch.stack(batch).to(device)
        else:
            input_tensor = batch.to(device)

        # 2. Dynamic K Calculation (Fix for index out of range)
        num_bank_items = self.bank_feats.shape[0]
        # Since data is sparse, we don't want k to be too big, 
        # or we will average in very far away points.
        k = min(20, num_bank_items) 
        
        # Safety fallback if bank is empty
        if k == 0:
            return [[0.0, 0.0]] * len(batch)

        with torch.no_grad():
            # Get features
            test_feats = self.model.get_features(input_tensor)
            test_feats = F.normalize(test_feats, p=2, dim=1)
            
            # Calculate Similarity (Dot Product)
            # Shape: [Batch_Size, Num_Bank_Items]
            sim_matrix = torch.mm(test_feats, self.bank_feats.t())
            
            # Get Top K matches
            # topk_sims: [Batch, k], topk_indices: [Batch, k]
            topk_sims, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
            
            # Retrieve coordinates
            neighbor_coords = self.bank_coords[topk_indices]
            
            # --- IMPROVEMENT: WEIGHTED AVERAGE ---
            # If data is sparse, simple mean is bad. We use Softmax on similarities.
            # Temperature (0.1) makes the weighting "sharper" (favors the top 1-2 matches more).
            temperature = 0.1
            weights = F.softmax(topk_sims / temperature, dim=1) 
            
            # Unsqueeze weights to match coords shape: [Batch, k, 1]
            weights = weights.unsqueeze(-1)
            
            # Weighted Sum: Sum(Coord * Weight)
            pred_norm = (neighbor_coords * weights).sum(dim=1)
            
            # Denormalize
            lat_pred = pred_norm[:, 0] * self.lat_std + self.lat_mean
            lon_pred = pred_norm[:, 1] * self.lon_std + self.lon_mean
            
            final_preds = torch.stack([lat_pred, lon_pred], dim=1)

        return final_preds.cpu().tolist()

def get_model() -> Model:
    return Model()