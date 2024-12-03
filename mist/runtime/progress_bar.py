"""Progress bars for MIST training and validation loops."""
from rich.progress import (
    BarColumn,
    TextColumn,
    Progress,
    MofNCompleteColumn,
    TimeElapsedColumn
)
import numpy as np

class TrainProgressBar(Progress):
    def __init__(self, current_epoch, fold, epochs, train_steps):
        super().__init__()

        epoch_width = len(str(epochs))
        self.progress = Progress(
            TextColumn(f"Fold {fold}: Epoch{current_epoch: {epoch_width}}/{epochs}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[loss]}"),
            TextColumn("•"),
            TextColumn("{task.fields[lr]}"),  # Learning rate column
        )

        # Initialize tasks with loss and learning rate fields
        self.task = self.progress.add_task(
            description="Training (loss)",
            total=train_steps,
            loss=f"loss: ",
            lr=f"lr: ",
        )

    def update(self, loss, lr):
        # Update loss task.
        self.progress.update(
            self.task,
            advance=1,
            loss=f"loss: {loss:.4f}",
            lr=f"lr: {np.format_float_scientific(lr, precision=3)}",
        )

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.stop()


class ValidationProgressBar(Progress):
    def __init__(self, val_steps):
        super().__init__()

        self.progress = Progress(
            TextColumn("Validating"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[loss]}"),
        )

        self.task = self.progress.add_task(
            description="Validation",
            total=val_steps,
            loss=f"val_loss: "
        )

    def update(self, loss):
        self.progress.update(self.task, advance=1, loss=f"val_loss: {loss:.4f}")

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.stop()