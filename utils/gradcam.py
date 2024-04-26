import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt


def generate_gradcam(model, target_layers, images, labels, rgb_imgs):
    results = []
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    for image, label, np_image in zip(images, labels, rgb_imgs):
        targets = [ClassifierOutputTarget(label.item())]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(
            input_tensor=image.unsqueeze(0), targets=targets, aug_smooth=True
        )

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            np_image / np_image.max(), grayscale_cam, use_rgb=True
        )
        results.append(visualization)
    return results


def visualize_gradcam(misimgs, mistgts, mispreds, classes):
    fig, axes = plt.subplots(len(misimgs) // 2, 2)
    fig.tight_layout()
    for ax, img, tgt, pred in zip(axes.ravel(), misimgs, mistgts, mispreds):
        ax.imshow(img)
        ax.set_title(f"{classes[tgt]} | {classes[pred]}")
        ax.grid(False)
        ax.set_axis_off()
    plt.show()

def plot_gradcam(model, data, classes, target_layers, number_of_samples, inv_normalize=None, targets=None, transparency = 0.60, figsize=(10,10), rows=2, cols=5):
    
    fig = plt.figure(figsize=figsize)
    
    cam = GradCAM(model=model, target_layers=target_layers)
    for i in range(number_of_samples):
            plt.subplot(rows, cols, i + 1)
            input_tensor = data[i][0]
    
            # Get the activations of the layer for the images
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
    
            # Get back the original image
            img = input_tensor.squeeze(0).to('cpu')
            if inv_normalize is not None:
                img = inv_normalize(img)
            rgb_img = np.transpose(img, (1, 2, 0))
            rgb_img = rgb_img.numpy()
    
            # Mix the activations on the original image
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)
    
            # Display the images on the plot
            plt.imshow(visualization)
            plt.title(f"Label: {classes[data[i][1].item()]} \n Prediction: {classes[data[i][2].item()]}")
            plt.xticks([])
            plt.yticks([])
