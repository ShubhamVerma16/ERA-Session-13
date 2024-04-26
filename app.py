import gradio as gr
import random
import numpy as np
from PIL import Image
import torch
import torchvision

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.resnet_lightning import ResNet
from utils.data import CIFARDataModule
from utils.transforms import test_transform
from utils.common import get_misclassified_data

inv_normalize = torchvision.transforms.Normalize(
    mean=[-0.50 / 0.23, -0.50 / 0.23, -0.50 / 0.23], std=[1 / 0.23, 1 / 0.23, 1 / 0.23]
)

datamodule = CIFARDataModule()
datamodule.setup()
classes = datamodule.train_dataset.classes

model = ResNet.load_from_checkpoint("model.ckpt")
model = model.to("cpu")

prediction_image = None


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def read_image(path):
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data


def sample_images():
    images = []
    length = len(datamodule.test_dataset)
    classes = datamodule.train_dataset.classes
    for i in range(10):
        idx = random.randint(0, length - 1)
        image, label = datamodule.test_dataset[idx]
        image = inv_normalize(image).permute(1, 2, 0).numpy()
        images.append((image, classes[label]))
    return images


def get_misclassified_images(misclassified_count):
    misclassified_images = []
    misclassified_data = get_misclassified_data(
        model=model,
        device="cpu",
        test_loader=datamodule.test_dataloader(),
        count=misclassified_count,
    )
    for i in range(misclassified_count):
        img = misclassified_data[i][0].squeeze().to("cpu")
        img = inv_normalize(img)
        img = np.transpose(img.numpy(), (1, 2, 0))
        label = f"Label: {classes[misclassified_data[i][1].item()]} | Prediction: {classes[misclassified_data[i][2].item()]}"
        misclassified_images.append((img, label))
    return misclassified_images


def get_gradcam_images(gradcam_layer, gradcam_count, gradcam_opacity):
    gradcam_images = []
    if gradcam_layer == "Layer1":
        target_layers = [model.layer1[-1]]
    elif gradcam_layer == "Layer2":
        target_layers = [model.layer2[-1]]
    else:
        target_layers = [model.layer3[-1]]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    data = get_misclassified_data(
        model=model,
        device="cpu",
        test_loader=datamodule.test_dataloader(),
        count=gradcam_count,
    )
    for i in range(gradcam_count):
        input_tensor = data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        # Get back the original image
        img = input_tensor.squeeze(0).to("cpu")
        if inv_normalize is not None:
            img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        # Mix the activations on the original image
        visualization = show_cam_on_image(
            rgb_img, grayscale_cam, use_rgb=True, image_weight=gradcam_opacity
        )
        label = f"Label: {classes[data[i][1].item()]} | Prediction: {classes[data[i][2].item()]}"
        gradcam_images.append((visualization, label))
    return gradcam_images


def show_hide_misclassified(status):
    if not status:
        return {misclassified_count: gr.update(visible=False)}
    return {misclassified_count: gr.update(visible=True)}


def show_hide_gradcam(status):
    if not status:
        return [gr.update(visible=False) for i in range(3)]
    return [gr.update(visible=True) for i in range(3)]


def set_prediction_image(evt: gr.SelectData, gallery):
    global prediction_image
    if isinstance(gallery[evt.index], dict):
        prediction_image = gallery[evt.index]["name"]
    else:
        prediction_image = gallery[evt.index][0]["name"]


def predict(
    is_misclassified,
    misclassified_count,
    is_gradcam,
    gradcam_count,
    gradcam_layer,
    gradcam_opacity,
    num_classes,
):
    misclassified_images = None
    if is_misclassified:
        misclassified_images = get_misclassified_images(int(misclassified_count))

    gradcam_images = None
    if is_gradcam:
        gradcam_images = get_gradcam_images(
            gradcam_layer, int(gradcam_count), gradcam_opacity
        )

    img = read_image(prediction_image)
    image_transformed = test_transform(image=img)["image"]
    output = model(image_transformed.unsqueeze(0))
    preds = torch.softmax(output, dim=1).squeeze().detach().numpy()
    indices = (
        output.argsort(descending=True).squeeze().detach().numpy()[: int(num_classes)]
    )
    predictions = {classes[i]: round(float(preds[i]), 2) for i in indices}

    return {
        miscalssfied_output: gr.update(value=misclassified_images),
        gradcam_output: gr.update(value=gradcam_images),
        prediction_label: gr.update(value=predictions),
    }


with gr.Blocks() as app:
    gr.Markdown("## ERA Session13 - CIFAR10 Classification with ResNet")
    with gr.Row():
        with gr.Column():
            with gr.Box():
                is_misclassified = gr.Checkbox(
                    label="Misclassified Images", info="Display misclassified images?"
                )
                misclassified_count = gr.Dropdown(
                    choices=["10", "20"],
                    label="Select Number of Images",
                    info="Number of Misclassified images",
                    visible=False,
                    interactive=True,
                )
                is_misclassified.input(
                    show_hide_misclassified,
                    inputs=[is_misclassified],
                    outputs=[misclassified_count],
                )
            with gr.Box():
                is_gradcam = gr.Checkbox(
                    label="GradCAM Images",
                    info="Display GradCAM images?",
                )
                gradcam_count = gr.Dropdown(
                    choices=["10", "20"],
                    label="Select Number of Images",
                    info="Number of GradCAM images",
                    interactive=True,
                    visible=False,
                )
                gradcam_layer = gr.Dropdown(
                    choices=["Layer1", "Layer2", "Layer3"],
                    label="Select the layer",
                    info="Please select the layer for which the GradCAM is required",
                    interactive=True,
                    visible=False,
                )
                gradcam_opacity = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.6,
                    label="Opacity",
                    info="Opacity of GradCAM output",
                    interactive=True,
                    visible=False,
                )

                is_gradcam.input(
                    show_hide_gradcam,
                    inputs=[is_gradcam],
                    outputs=[gradcam_count, gradcam_layer, gradcam_opacity],
                )
            with gr.Box():
                # file_output = gr.File(file_types=["image"])
                with gr.Group():
                    upload_gallery = gr.Gallery(
                        value=None,
                        label="Uploaded images",
                        show_label=False,
                        elem_id="gallery_upload",
                        columns=5,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                    )
                    upload_button = gr.UploadButton(
                        "Click to Upload images",
                        file_types=["image"],
                        file_count="multiple",
                    )
                    upload_button.upload(upload_file, upload_button, upload_gallery)

                with gr.Group():
                    sample_gallery = gr.Gallery(
                        value=sample_images,
                        label="Sample images",
                        show_label=True,
                        elem_id="gallery_sample",
                        columns=5,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                    )

                upload_gallery.select(set_prediction_image, inputs=[upload_gallery])
                sample_gallery.select(set_prediction_image, inputs=[sample_gallery])

            with gr.Box():
                num_classes = gr.Dropdown(
                    choices=[str(i + 1) for i in range(10)],
                    label="Select Number of Top Classes",
                    info="Number of Top target classes to be shown",
                )
            run_btn = gr.Button()
        with gr.Column():
            with gr.Box():
                miscalssfied_output = gr.Gallery(
                    value=None, label="Misclassified Images", show_label=True
                )
            with gr.Box():
                gradcam_output = gr.Gallery(
                    value=None, label="GradCAM Images", show_label=True
                )
            with gr.Box():
                prediction_label = gr.Label(value=None, label="Predictions")

        run_btn.click(
            predict,
            inputs=[
                is_misclassified,
                misclassified_count,
                is_gradcam,
                gradcam_count,
                gradcam_layer,
                gradcam_opacity,
                num_classes,
            ],
            outputs=[miscalssfied_output, gradcam_output, prediction_label],
        )


app.launch(server_name="0.0.0.0", server_port=9998)
