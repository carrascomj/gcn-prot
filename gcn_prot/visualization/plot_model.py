"""Visualization about the neural network model."""

import matplotlib.pyplot as plt


def plot_epoch(epochs, train_loss, test_loss, epoch):
    """Loss plot, neon colors."""
    # Other Neon colors of the palette -> #13CA91 #3B27BA #E847AE #FF9472
    plt.plot(epochs, train_loss, color="#13CA91", marker="o", label="Training loss")
    plt.plot(epochs, test_loss, color="#E847AE", marker="o", label="Validation loss")
    plt.title(
        f"Epoch {epoch}\n"
        f"Train Loss: {train_loss[-1]:.3f}\n"
        f"Val. Loss: {test_loss[-1]:.3f}"
    )
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.show()
