import torch.nn as nn
import torch

class PneumoniaModel(nn.Module):
    def __init__(self, base_model):
        super(PneumoniaModel, self).__init__()
     #   self.image_encoder = base_model.image_encoder
        self.base_model = base_model
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Mean pooling
        self.classifier = nn.Linear(1024, 1)  # Linear layer for binary classification

    def forward(self, images):
        # Extract features
    #    features = self.image_encoder(images)  # Output: (32, 1024)
        features = self.base_model.encode_image(images)  # Output: (32, 1024)
#        pooled_features = self.pooling(features.T).squeeze()  # Shape: (1024,)
        logits = self.classifier(features).squeeze()   # Shape: (1,)
        return logits
