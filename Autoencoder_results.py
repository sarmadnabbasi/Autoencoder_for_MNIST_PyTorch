import torch
import matplotlib.pyplot as plt
import numpy as np

from autoencoder_class import AutoEncoder_MNIST
from utils import load_data, load_model, select_torch_device

device = select_torch_device()

def solution_2_1():
    ## (a) Plot the learning curve (𝐽𝑀 vs epoch) of training
    #      and test dataset for M=2
    file_path = 'J_M_Autoencoder.csv'
    J_M_data_AE = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    print("(a) Plot the learning curve (𝐽𝑀 vs epoch) of "
          "training and test dataset for M=2")
    plt.figure("Solution 2.1 (a)")
    plt.plot(J_M_data_AE[:, 0], label='M_2_train')
    plt.plot(J_M_data_AE[:, 1], label='M_2_test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Solution 2.1 (a) Learning curve Jm vs epochs")
    plt.show()

    ## (b) Plot the learning curve (𝐽𝑀 vs epoch) of training and test dataset.
    # PCA results
    M5 = 0.0444937002720817
    M10 = 0.03405504342385595

    print("(b) J_M of Autoencoder and PCA")
    plt.figure("Solution 2.1 (b)")
    plt.plot(J_M_data_AE[:, 2], label='M_5_Autoencoder')
    plt.plot(J_M_data_AE[:, 4], label='M_10_Autoencoder')
    plt.plot(np.full((20), M5), label='M_5_PCA')
    plt.plot(np.full((20), M10), label='M_10_PCA')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Solution 2.1 (a) ). Plot J_M of the autoencoder and PCA together.")
    plt.show()

def solution_2_2():
    ## Loading Model
    M = 2
    model_folder = "models"
    model_name = "model_M_" + str(M)  # M = 2 ("model_M_2") ,
    # M= 5 ("model_M_5"), M= 10 ("model_M_10")
    model = AutoEncoder_MNIST(m=M).to(device)
    load_model(model_folder=model_folder,
               model_name=model_name, model=model)

    ## Load Data
    batch_size = 256
    train_loader, val_loader = load_data(batch_size=batch_size)

    # Select 100 samples for each class (digits 0, 1, and 9)
    selected_images = []
    selected_labels = []

    for digit in [0, 1, 9]:
        images, labels = [], []
        for batch in train_loader:
            img, lbl = batch
            # Filter out the images of the selected digit
            mask = lbl == digit
            images.append(img[mask])
            labels.append(lbl[mask])

            # If we have 100 samples, break the loop
            if len(images) * len(images[0]) >= 100:
                break
        # Get first 100 images of the digit and labels
        selected_images.append(torch.cat(images)[:100])
        selected_labels.append(torch.cat(labels)[:100])

    # Stack selected images for each class
    selected_images = torch.cat(selected_images, dim=0)
    selected_labels = torch.cat(selected_labels, dim=0)

    # Obtain latent representations (z) using the encoder
    with torch.no_grad():
        dump, latent_representations = model(selected_images.view(-1, 28 * 28).to(device))

    # Convert latent representations to numpy array for plotting
    latent_representations = latent_representations.cpu().numpy()

    # Plot the 2D latent space
    plt.figure("Solution 2.2 (a)  Plot latent space in 2D space  for each class",
               figsize=(8, 6))

    # Plot each class with a different color
    colors = ['blue', 'brown', 'darkturquoise']
    labels = ['Digit 0', 'Digit 1', 'Digit 9']

    for i, digit in enumerate([0, 1, 9]):
        indices = selected_labels == digit
        plt.scatter(latent_representations[indices, 0],
                    latent_representations[indices, 1], label=labels[i],
                    c=colors[i], alpha=0.5)

    # Add title and labels
    plt.title('Solution 2.2 (a) 2D Latent Space of MNIST (Autoencoder)')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    solution_2_1()
    solution_2_2()