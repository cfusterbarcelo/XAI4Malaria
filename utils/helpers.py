# utils/helpers.py

class EarlyStopping:
    def __init__(self, patience=10, mode="max", delta=0.0):
        """
        Args:
            patience (int): Number of epochs to wait after no improvement.
            mode (str): 'min' for loss, 'max' for accuracy.
            delta (float): Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "min" and score > self.best_score - self.delta) or \
             (self.mode == "max" and score < self.best_score + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
