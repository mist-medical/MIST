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
