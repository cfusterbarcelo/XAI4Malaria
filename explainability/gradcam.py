# explainability/gradcam.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.gradients = None
        self.activations = None

        # Hook the target layer
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward()

        # Get mean of gradients along channel dimension (global average pooling)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # shape: [B, C, 1, 1]

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU + normalization
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, input_tensor.shape[2:][::-1])  # resize to input size
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam  # shape: [H, W]

