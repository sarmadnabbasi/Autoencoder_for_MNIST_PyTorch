import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # For plotting graphs during debug

from Autoencoder_results import solution_2_1, solution_2_2
from autoencoder_class import AutoEncoder_MNIST
from utils import load_data, save_model, display_result, select_torch_device

device = select_torch_device()

def train(train_data, val_data, num_epochs, loss_fn, opt):
    train_loss_history = []  # To store training loss for each epoch
    val_loss_history = []  # To store validation loss for each epoch
    J_M_train = []  # To store mean squared error for training
    J_M_val = []  # To store mean squared error for validation

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        msd_train = 0
        for batch in train_data:
            images, _ = batch
            images = images.view(-1, 28 * 28).to(device)
            # Forward pass
            reconstructed, _ = model(images)
            loss = loss_fn(reconstructed, images)
            # Backward pass and optimization
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train_loss += loss.item()

            with torch.no_grad():
                input = images.cpu().detach().numpy()
                target = reconstructed.cpu().detach().numpy()
                squared_diff = (input - target) ** 2
                msd_train += np.mean(squared_diff)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_msd = msd_train / len(train_loader)
        train_loss_history.append(avg_train_loss)
        J_M_train.append(avg_train_msd)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        msd_val = 0
        with torch.no_grad():
            for batch in val_data:
                images, _ = batch
                images = images.view(-1, 28 * 28).to(device)
                # Forward pass
                reconstructed, _ = model(images)
                loss = loss_fn(reconstructed, images)
                total_val_loss += loss.item()

                input = images.cpu().detach().numpy()
                target = reconstructed.cpu().detach().numpy()
                squared_diff = (input - target) ** 2
                msd_val += np.mean(squared_diff)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_msd = msd_val / len(val_loader)
        val_loss_history.append(avg_val_loss)
        J_M_val.append(avg_val_msd)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f},"
              f" Train MSD: {avg_train_msd:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f},"
              f" Val MSD: {avg_val_msd:.4f}")

    return (train_loss_history, val_loss_history,
            J_M_train, J_M_val)

if __name__ == "__main__":
    ## Hyperparameters
    batch_size = 256
    learning_rate = 0.001
    epochs = 20
    M = 2

    ## Load Data
    train_loader, val_loader = load_data(batch_size=batch_size)

    ## Initialize Model
    model = AutoEncoder_MNIST(m=M).to(device)
    print(model)
    loss = nn.MSELoss()  # Reconstruction error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ## Training
    (train_loss, val_loss,
     train_msd, val_msd) = train(train_loader,val_loader,
                                 num_epochs=epochs,loss_fn = loss,
                                 opt = optimizer)
    print(train_loss)
    print(val_loss)

    ## (a) Plot the learning curve (𝐽𝑀 vs epoch) of training and test dataset
    print("(a) Plot the learning curve (𝐽𝑀 vs epoch) of training and test dataset.")
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    ## Save Model
    save_model(model_folder="models",
               model_name="model_M_"+str(M),
               model=model)

    ## Display results
    dataiter = iter(val_loader)
    images, labels = next(dataiter)

    images_flatten = images.view(images.size(0), -1).to(device)
    output, _ = model(images_flatten)
    images = images.numpy()
    output = output.view(batch_size, 1, 28, 28)
    output = output.detach().cpu().numpy()

    display_result(images, output)

    solution_2_1()
    solution_2_2()