import os
import torchvision.utils as vutils
from PIL import Image


def save_images(images, path):
    """
    Save a batch of images as a grid to the specified path.
    Args:
        images (torch.Tensor): Batch of images to save.
        path (str): File path to save the image grid.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = (images + 1) / 2.0  # Rescale images from [-1, 1] to [0, 1]
    grid = vutils.make_grid(images, normalize=True, nrow=8)
    vutils.save_image(grid, path)


def ensure_dir(directory):
    """
    Ensure a directory exists. If it does not, create it.
    Args:
        directory (str): Path of the directory to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
