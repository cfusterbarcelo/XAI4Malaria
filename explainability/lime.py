import os
import cv2
import numpy as np
import torch
from lime import lime_image
from PIL import Image
from utils.xai_utils import save_cam

class LimeExplainer:
    """
    LIME explainer for image models.

    Follows our SHAP-style API but uses LIME to compute super-pixel attributions.
    Reconstructs a continuous heatmap from LIME’s segment weights and saves both heatmap
    and overlay via `save_cam` from xai_utils.
    """
    def __init__(self, model, transform, output_dir, num_samples=1000, batch_size=1):
        """
        Args:
            model: PyTorch model for classification (expects batch tensor input).
            transform: preprocessing transform (PIL -> tensor + normalization).
            output_dir: base directory to save results (e.g., blabla/fold1/lime/).
            num_samples: number of perturbed samples for LIME.
            batch_size: placeholder (LIME uses per-instance calls).
        """
        self.model = model
        self.transform = transform
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        # prepare output subdirs
        os.makedirs(os.path.join(output_dir, 'heatmap'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'overlay'), exist_ok=True)
        # instantiate LIME image explainer
        self.explainer = lime_image.LimeImageExplainer()

    def _batch_predict(self, images):
        """
        Prediction function for LIME: takes a list of H×W×3 arrays, returns NxK probs.
        """
        batch = []
        for img in images:
            pil = Image.fromarray(img.astype('uint8'), 'RGB')
            x = self.transform(pil)
            batch.append(x)
        batch_tensor = torch.stack(batch).to(next(self.model.parameters()).device)
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def explain_image(self, img_path, label=None, hide_color=0, num_features=5, top_labels=1):
        """
        Generate and save LIME explanations for a single image.

        Args:
            img_path: str, path to input image file.
            label: int or None, used as "true" label in filenames.
            hide_color: int, pixel value for masked super-pixels.
            num_features: int, number of super-pixels to highlight.
            top_labels: int, how many top predicted classes to explain.
        """
        # load image
        orig_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(orig_pil)

        # run LIME
        explanation = self.explainer.explain_instance(
            image=img_np,
            classifier_fn=self._batch_predict,
            hide_color=hide_color,
            num_samples=self.num_samples,
            top_labels=top_labels
        )

        # segments array: each pixel labeled with its super-pixel id
        segments = explanation.segments

        # for each predicted label, reconstruct continuous attribution map
        for class_id in explanation.top_labels:
            # local explanation: list of (segment_id, weight)
            weights = dict(explanation.local_exp[class_id])
            # build attribution map
            cam = np.zeros_like(segments, dtype=float)
            for seg_id, weight in weights.items():
                cam[segments == seg_id] = weight

            base = os.path.splitext(os.path.basename(img_path))[0]
            filename = f"{base}.png"

            # save heatmap and overlay
            heat_dir = os.path.join(self.output_dir, 'heatmap')
            ovl_dir  = os.path.join(self.output_dir, 'overlay')

            save_cam(
                cam=cam,
                orig_rgb=img_np,
                out_dir=heat_dir,
                filename=filename,
                true_label=label if label is not None else -1,
                pred_label=class_id,
                mode='heatmap',
                colormap=cv2.COLORMAP_JET
            )
            save_cam(
                cam=cam,
                orig_rgb=img_np,
                out_dir=ovl_dir,
                filename=filename,
                true_label=label if label is not None else -1,
                pred_label=class_id,
                mode='overlay',
                colormap=cv2.COLORMAP_JET
            )
