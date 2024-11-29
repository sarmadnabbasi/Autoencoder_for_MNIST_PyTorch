from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import os

def load_data(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))  # to the range [0, 1]
    ])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    mnist_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    train_size = int(0.8 * len(mnist_full))  # 80% for training
    val_size = len(mnist_full) - train_size  # Remaining for validation

    train_dataset, val_dataset = random_split(mnist_full, [train_size, val_size])

    mnist_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    mnist_loader_test = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return mnist_loader_train, mnist_loader_test

def save_model(model_folder, model_name, model):
    MODEL_PATH = Path(model_folder)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = model_name+".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving Model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)

def load_model(model_folder, model_name, model):
    MODEL_PATH = Path(model_folder)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = model_name+".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True))

def display_result(images, output):
    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

def select_torch_device():
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        device = "cpu"
        print(f"Device name: {device}")
    return device


