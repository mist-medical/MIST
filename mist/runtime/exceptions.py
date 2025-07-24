"""Exceptions for MIST."""


class InsufficientValidationSetError(Exception):
    """Raised if validation set size is smaller than the number of GPUs."""
    def __init__(self, val_size: int, world_size: int) -> None:
        self.message = (
            f"Validation set size of {val_size} is too small for {world_size} "
            "GPUs. Please increase the validation set size or reduce the "
            "number of GPUs."
        )
        super().__init__(self.message)


class NaNLossError(Exception):
    """Raised if the loss is NaN."""
    def __init__(self, epoch) -> None:
        self.message = (
            f"Encountered NaN loss value in epoch {epoch}. Stopping training. "
            "Consider using a different optimizer, reducing the learning rate, "
            "or using gradient clipping."
        )
        super().__init__(self.message)


class NoGPUsAvailableError(Exception):
    """Raised if no GPU is available."""
    def __init__(self) -> None:
        self.message = (
            "No GPU available. Please check your hardware configuration."
        )
        super().__init__(self.message)
