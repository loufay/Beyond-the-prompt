import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedPromptEnsemble(nn.Module):
    def __init__(self, num_prompts_no_finding, num_prompts_pneumonia, embedding_dim, learn_temperature=False):
        super(WeightedPromptEnsemble, self).__init__()
        # Learnable weights for each template in both classes
        self.weights_no_finding = nn.Parameter(torch.randn(num_prompts_no_finding))
        self.weights_pneumonia = nn.Parameter(torch.randn(num_prompts_pneumonia))

        # Temperature setting
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(0.15))  # Trainable parameter
        else:
            self.register_buffer("temperature", torch.tensor(0.1))  # Fixed parameter, not updated by optimizer

    

    def forward(self, no_finding_templates, pneumonia_templates):
        """
        pneumonia_templates: Tensor of shape (num_prompts_pneumonia, embedding_dim)
        no_finding_templates: Tensor of shape (num_prompts_no_finding, embedding_dim)
        """
        # Normalize weights using softmax
        # temperature = 0.1 # more confident weighting
        temp = F.softplus(self.temperature) + 1e-6 if self.learn_temperature else self.temperature

        alpha_no_finding = F.softmax(self.weights_no_finding/temp, dim=0)
        alpha_pneumonia = F.softmax(self.weights_pneumonia/temp, dim=0)

        self.alpha_no_finding = alpha_no_finding
        self.alpha_pneumonia = alpha_pneumonia

        # Weighted sum of template embeddings
        ensemble_no_finding = torch.einsum('nd,n->d', no_finding_templates, alpha_no_finding)
        ensemble_pneumonia = torch.einsum('nd,n->d', pneumonia_templates, alpha_pneumonia)

        return ensemble_no_finding,ensemble_pneumonia


class ZeroShotClassifier(nn.Module):
    def __init__(self, num_prompts_no_finding, num_prompts_pneumonia, embedding_dim, learn_temperature=False):
        super(ZeroShotClassifier, self).__init__()
        self.ensemble = WeightedPromptEnsemble(num_prompts_no_finding, num_prompts_pneumonia,  embedding_dim, learn_temperature)

    def forward(self, image_embeddings, no_finding_templates, pneumonia_templates):
        # Get weighted ensemble for both classes
        ensemble_no_finding,ensemble_pneumonia = self.ensemble(no_finding_templates, pneumonia_templates)

        # Compute cosine similarity between image(s) and each class ensemble
        sim_no_finding = F.cosine_similarity(image_embeddings, ensemble_no_finding.unsqueeze(0), dim=-1)
        sim_pneumonia = F.cosine_similarity(image_embeddings, ensemble_pneumonia.unsqueeze(0), dim=-1)

        # Stack similarities as logits
        logits = torch.stack([sim_no_finding, sim_pneumonia], dim=1)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

    def initialize_weights(self, no_finding_templates, pneumonia_templates):
        # Initialize weights to uniform distribution
        self.ensemble.weights_no_finding.data.fill_(1.0 / no_finding_templates.size(0))
        self.ensemble.weights_pneumonia.data.fill_(1.0 / pneumonia_templates.size(0))
        
        # 
    def entropy_loss(self, probabilities):
        # Entropy loss to encourage confident predictions
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        return entropy.mean()
    
    def kl_divergence(self, probabilities):
        target_probabilities = torch.full_like(probabilities, 0.5).to(probabilities.device)
        # KL divergence loss between predicted and target probabilities
        kl_div = -F.kl_div(F.log_softmax(probabilities, dim=1), target_probabilities, reduction='batchmean')
        return kl_div


# Prepare samples
def prepare_samples(df, feature_columns,disease):
    return {
        "img_name": df["Path"].tolist(),
        "labels": df[disease].tolist(),
        "features": [np.array(row) for row in df[feature_columns].values]
    }


