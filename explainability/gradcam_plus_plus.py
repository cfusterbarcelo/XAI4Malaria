# gradcam_plus_plus.py
# Implementation of GradCAM++ for PyTorch models, mirroring the structure of gradcam.py

import torch
import torch.nn.functional as F

class GradCAMPlusPlus:
    """
    Compute GradCAM++ saliency maps for convolutional neural networks.
    Usage:
        # Pass either the model + layer name string, or the module object itself:
        tgt_module = dict(model.named_modules())["layer4.2.conv3"]
        cam_generator = GradCAMPlusPlus(model, tgt_module)
        # or
        cam_generator = GradCAMPlusPlus(model, "layer4.2.conv3")
        heatmap = cam_generator(input_tensor, class_idx)
    """
    def __init__(self, model: torch.nn.Module, target_layer):
        self.model = model.eval()
        self.gradients = None
        self.activations = None

        # Resolve target layer: accept string name or Module object
        if isinstance(target_layer, str):
            modules = dict(self.model.named_modules())
            if target_layer not in modules:
                raise KeyError(
                    f"Layer '{target_layer}' not found in model.named_modules()."
                    f" Available layers: {list(modules.keys())}"
                )
            layer = modules[target_layer]
        elif isinstance(target_layer, torch.nn.Module):
            layer = target_layer
        else:
            raise TypeError(
                "target_layer must be a string name or a torch.nn.Module instance"
            )

        # Register forward/backward hooks
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        # Forward pass
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Compute score and gradients
        score = logits[:, class_idx].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients         # [B, C, H, W]
        acts  = self.activations       # [B, C, H, W]

        # GradCAM++ computations
        grads_pow2 = grads.pow(2)
        grads_pow3 = grads.pow(3)
        sum_acts = acts.pow(2).sum(dim=(2,3), keepdim=True)
        epsilon = 1e-8

        alpha_num = grads_pow2
        alpha_den = 2 * grads_pow2 + (acts * grads_pow3).sum(dim=(2,3), keepdim=True)
        alphas = alpha_num / (alpha_den + epsilon)

        weights = (F.relu(grads) * alphas).sum(dim=(2,3))  # [B, C]
        cam = F.relu((weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=1))  # [B, H, W]

        # Normalize per map
        b, h, w = cam.shape
        cam_flat = cam.view(b, -1)
        cam_min = cam_flat.min(dim=1)[0].view(b,1,1)
        cam_max = cam_flat.max(dim=1)[0].view(b,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + epsilon)

        # Upsample to input size
        cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze(1)  # [B, H, W]
