import time
from deeplib.logging.plotting.plotter import Plotter
from deeplib.logging.progress.utils import disp_len

total_epochs = 100
total_batches = 10

values = []

plotter = Plotter()


for epoch in range(total_epochs):
    for batch in range(total_batches):
        pass

    plotter.add_scalars(
        "test", {"train_loss": 0.01 * epoch, "val_loss": 0.8 - 0.01 * epoch}
    )
    plotter.add_scalars(
        "test2", {"train_loss": 0.01 * epoch, "val_loss": 0.8 - 0.01 * epoch}
    )

    time.sleep(1)

    values.append(0.01 * epoch)
