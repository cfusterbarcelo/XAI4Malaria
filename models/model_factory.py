# models/model_factory.py

from models.spcnn import SPCNN

model_registry = {
    "spcnn": SPCNN,
    # future: "resnet18": torchvision.models.resnet18,
}

def get_model(model_config):
    """
    Dynamically loads and returns a model instance based on the configuration.

    Args:
        model_config (dict): Dictionary with model parameters:
            - name (str): model name (e.g., 'spcnn')
            - input_shape (tuple): image shape (e.g., (3, 32, 32))
            - num_classes (int): number of output classes (e.g., 2)

    Returns:
        nn.Module: PyTorch model instance
    """
    name = model_config.get("name").lower()
    input_shape = model_config.get("input_shape", (3, 32, 32))
    num_classes = model_config.get("num_classes", 2)

    if name == "spcnn":
        from models.spcnn import SPCNN
        return SPCNN(input_shape=input_shape, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model name: {name}")
