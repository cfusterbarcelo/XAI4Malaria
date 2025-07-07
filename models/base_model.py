# models/base_model.py

import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all classification models.

    Subclasses must implement:
        - self.feature_extractor
        - self.classifier
    """

    def __init__(self, input_shape=(3, 32, 32), num_classes=2):
        """
        Args:
            input_shape (tuple): Input image shape, e.g., (3, 32, 32)
            num_classes (int): Number of output classes (binary classification = 2)
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Placeholder modules to be defined in subclasses
        self.feature_extractor = None
        self.classifier = None

        self._build()  # Let subclass build internal architecture

    @abstractmethod
    def _build(self):
        """
        Build the feature extractor and classifier.
        Must be implemented by all subclasses.
        """
        pass

    def forward(self, x):
        """
        Forward pass of the model.
        """
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out

    def extract_features(self, x):
        """
        Expose intermediate features for interpretability/XAI.
        """
        return self.feature_extractor(x)

    def get_input_shape(self):
        return self.input_shape

    def get_num_classes(self):
        return self.num_classes
