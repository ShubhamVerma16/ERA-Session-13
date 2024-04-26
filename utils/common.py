import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torchvision
from torchinfo import summary
from torch_lr_finder import LRFinder


def find_lr(model, optimizer, criterion, device, trainloader, numiter, startlr, endlr):
    lr_finder = LRFinder(
        model=model, optimizer=optimizer, criterion=criterion, device=device
    )

    lr_finder.range_test(
        train_loader=trainloader,
        start_lr=startlr,
        end_lr=endlr,
        num_iter=numiter,
        step_mode="exp",
    )

    lr_finder.plot()

    lr_finder.reset()


def one_cycle_lr(optimizer, maxlr, steps, epochs):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=maxlr,
        steps_per_epoch=steps,
        epochs=epochs,
        pct_start=5 / epochs,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy="linear",
    )
    return scheduler


def show_random_images_for_each_class(train_data, num_images_per_class=16):
    for c, cls in enumerate(train_data.classes):
        rand_targets = random.sample(
            [n for n, x in enumerate(train_data.targets) if x == c],
            k=num_images_per_class,
        )
        show_img_grid(np.transpose(train_data.data[rand_targets], axes=(0, 3, 1, 2)))
        plt.title(cls)


def show_img_grid(data):
    try:
        grid_img = torchvision.utils.make_grid(data.cpu().detach())
    except:
        data = torch.from_numpy(data)
        grid_img = torchvision.utils.make_grid(data)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))


def show_random_images(data_loader):
    data, target = next(iter(data_loader))
    show_img_grid(data)


def show_model_summary(model, batch_size):
    summary(
        model=model,
        input_size=(batch_size, 3, 32, 32),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        verbose=1,
    )


def lossacc_plots(results):
    plt.plot(results["epoch"], results["trainloss"])
    plt.plot(results["epoch"], results["testloss"])
    plt.legend(["Train Loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.show()

    plt.plot(results["epoch"], results["trainacc"])
    plt.plot(results["epoch"], results["testacc"])
    plt.legend(["Train Acc", "Validation Acc"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.show()


def lr_plots(results, length):
    plt.plot(range(length), results["lr"])
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs Epochs")
    plt.show()


def get_misclassified(model, testloader, device, mis_count=10):
    misimgs, mistgts, mispreds = [], [], []
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            misclassified = torch.argwhere(pred.squeeze() != target).squeeze()
            for idx in misclassified:
                if len(misimgs) >= mis_count:
                    break
                misimgs.append(data[idx])
                mistgts.append(target[idx])
                mispreds.append(pred[idx].squeeze())
    return misimgs, mistgts, mispreds


# def plot_misclassified(misimgs, mistgts, mispreds, classes):
#     fig, axes = plt.subplots(len(misimgs) // 2, 2)
#     fig.tight_layout()
#     for ax, img, tgt, pred in zip(axes.ravel(), misimgs, mistgts, mispreds):
#         ax.imshow((img / img.max()).permute(1, 2, 0).cpu())
#         ax.set_title(f"{classes[tgt]} | {classes[pred]}")
#         ax.grid(False)
#         ax.set_axis_off()
#     plt.show()

def get_misclassified_data(model, device, test_loader, count):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()

    # List to store misclassified Images
    misclassified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:

            # Migrate the data to the device
            data, target = data.to(device), target.to(device)

            # Extract single image, label from the batch
            for image, label in zip(data, target):

                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)

                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)

                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
            if len(misclassified_data) >= count:
                        break
            
    return misclassified_data[:count]

def plot_misclassified(data, classes, size=(10, 10), rows=2, cols=5, inv_normalize=None):
    fig = plt.figure(figsize=size)
    number_of_samples = len(data)
    for i in range(number_of_samples):
        plt.subplot(rows, cols, i + 1)
        img = data[i][0].squeeze().to('cpu')
        if inv_normalize is not None:
            img = inv_normalize(img)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(f"Label: {classes[data[i][1].item()]} \n Prediction: {classes[data[i][2].item()]}")
        plt.xticks([])
        plt.yticks([])

