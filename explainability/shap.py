# explainability/shap.py
"""
Module defining SHAP-based XAI for image classification.
Provides a unified interface to both DeepExplainer and GradientExplainer.
"""
import torch
import shap
import numpy as np

class SHAPExplainer:
    """
    Wrapper for SHAP explainers producing per-pixel attribution heatmaps.

    Args:
        model (torch.nn.Module): Trained classification model.
        background (torch.Tensor): Background samples [N,C,H,W] for SHAP.
        device (torch.device): Device to run on.
        explainer_type (str): 'deep' or 'gradient'.
        nsamples (int): Number of samples for SHAP value estimation.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        background: torch.Tensor,
        device: torch.device,
        explainer_type: str = 'gradient',
        nsamples: int = 100
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.nsamples = nsamples
        self.background = background.to(device)

        # Initialize SHAP explainer
        if explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, self.background)
        elif explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(
                (self.model, getattr(self.model, 'feature_extractor', None)),
                self.background
            )
        else:
            raise ValueError(f"Unknown explainer_type: {explainer_type}")

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        """
        Compute SHAP values for the given input batch.

        Args:
            input_tensor (torch.Tensor): Input images [B,C,H,W].
            class_idx (int): Index of the class to explain. Defaults to predicted argmax.

        Returns:
            torch.Tensor: Heatmaps [B,H,W] normalized to [0,1].
        """
        # Ensure input on device
        x = input_tensor.to(self.device)

        # Determine class index
        with torch.no_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).cpu().item()

        # Compute SHAP values
        shap_values = self.explainer.shap_values(x, nsamples=self.nsamples)

        # Select for class
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                sv_np = shap_values[0]
            else:
                sv_np = shap_values[class_idx]
        else:
            sv_np = shap_values

        # sv_np: numpy array of shape (B, H, W, C) or (B, C, H, W)
        # Convert to shape (B, C, H, W)
        if isinstance(sv_np, np.ndarray):
            if sv_np.ndim == 4:
                # detect channel position: assume last axis is channel
                if sv_np.shape[-1] in (1,3,  # grayscale or RGB
                                        input_tensor.shape[1]):
                    # (B,H,W,C) -> (B,C,H,W)
                    sv_np = np.transpose(sv_np, (0,3,1,2))
            # else if shape is already (B,C,H,W), leave
            sv_t = torch.from_numpy(sv_np).to(self.device)
        else:
            # if shap returns torch.Tensor
            sv_t = sv_np.to(self.device)
            if sv_t.ndim == 4 and sv_t.shape[1] not in (1,3):
                # maybe channels last? move last->1
                sv_t = sv_t.permute(0,3,1,2)

        # Aggregate over channels
        heatmaps = sv_t.abs().sum(dim=1)  # [B,H,W]

        # Normalize to [0,1]
        b, *rest = heatmaps.shape
        # rest should be [H,W]
        flat = heatmaps.view(b, -1)
        mins = flat.min(dim=1)[0].view(b,1,1)
        maxs = flat.max(dim=1)[0].view(b,1,1)
        normed = (heatmaps - mins) / (maxs - mins + 1e-8)

        return normed  # [B,H,W]
