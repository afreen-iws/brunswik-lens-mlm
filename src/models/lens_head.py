
import torch, torch.nn as nn, torch.nn.functional as F

class BaselineHead(nn.Module):
    def __init__(self, h_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.cls = nn.Linear(256, num_classes)
    def forward(self, h):
        return self.cls(self.mlp(h))

class LensHead(nn.Module):
    def __init__(self, h_dim, num_classes, validity_prior=None, lambda_validity=0.0):
        super().__init__()
        self.u_params = nn.Parameter(torch.zeros(3))  # utilization logits
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.cls = nn.Linear(256 + 3, num_classes)
        self.lambda_validity = lambda_validity
        if validity_prior is None: validity_prior = torch.zeros(3)
        self.register_buffer("validity_prior", validity_prior.float())

    def forward(self, h, z):
        u = torch.softmax(self.u_params, dim=0)           # (3,)
        fused = torch.cat([ self.mlp(h), z * u ], dim=1)  # weight cues
        return self.cls(fused), u

    def lens_regularizer(self):
        if self.lambda_validity <= 0:
            return torch.tensor(0.0, device=self.validity_prior.device)
        u = torch.softmax(self.u_params, dim=0)
        v = torch.softmax(self.validity_prior, dim=0)
        return F.mse_loss(u, v) * self.lambda_validity
