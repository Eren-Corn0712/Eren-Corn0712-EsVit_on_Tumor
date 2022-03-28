from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_augmentation(data_loader):
    for images, _ in data_loader:
        image = make_grid(images)
        show(image)
        plt.tight_layout()
        plt.show()


def plot_augmentation_muti(data_loader):
    for images, _ in data_loader:
        for image in images:
            image = make_grid(image)
            show(image)
        plt.tight_layout()
        plt.show()
