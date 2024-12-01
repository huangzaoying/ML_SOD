import io
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
from net.models.SUM import SUM
from net.configs.config_setting import setting_config
from utils.data_process_uni import preprocess_img, postprocess_img


def setup_model(device):
    config = setting_config
    model_cfg = config.model_config
    if config.network == "sum":
        model = SUM(
            num_classes=model_cfg["num_classes"],
            input_channels=model_cfg["input_channels"],
            depths=model_cfg["depths"],
            depths_decoder=model_cfg["depths_decoder"],
            drop_path_rate=model_cfg["drop_path_rate"],
        )
        model.load_state_dict(
            torch.load("net/pre_trained_weights/sum_model.pth", map_location=device)
            # torch.load(
            #     "/media/data/WWZ/HZY/ml_hw1/ML_SOD/best_model.pth",
            #     map_location=device,
            # )
        )
        print(
            "load model from",
            "/media/data/WWZ/HZY/ml_hw1/ML_SOD/best_model_1122_02.pth",
        )
        model.to(device)
        return model
    else:
        raise NotImplementedError(
            "The specified network configuration is not supported."
        )


def load_and_preprocess_image(img_path):
    img_padded, orig_img = preprocess_img(img_path, channels=3)
    orig_size = orig_img.shape[:2]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = transform(Image.fromarray(img_padded))
    return image, orig_size


def saliency_map_prediction(img_path, condition, model, device):
    img, orig_size = load_and_preprocess_image(img_path)
    img = img.unsqueeze(0).to(device)
    one_hot_condition = torch.zeros((1, 4), device=device)
    one_hot_condition[0, condition] = 1
    model.eval()
    with torch.no_grad():
        pred_saliency = model(img, one_hot_condition)

    restored_pred_saliency = postprocess_img(
        pred_saliency.squeeze().cpu().numpy(), img_path
    )
    return restored_pred_saliency, orig_size


def overlay_heatmap_on_image(original_img_path, heatmap_img_path, output_img_path):
    # Read the original image
    orig_image = cv2.imread(original_img_path)
    orig_size = orig_image.shape[:2]  # Height, Width

    # Read the heatmap image
    overlay_heatmap = cv2.imread(heatmap_img_path, cv2.IMREAD_GRAYSCALE)

    # Resize the heatmap to match the original image size
    overlay_heatmap = cv2.resize(overlay_heatmap, (orig_size[1], orig_size[0]))

    # Apply color map to the heatmap
    overlay_heatmap = cv2.applyColorMap(overlay_heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlay_image = cv2.addWeighted(orig_image, 1, overlay_heatmap, 0.8, 0)

    # Save the result
    cv2.imwrite(output_img_path, overlay_image)


def main():
    parser = argparse.ArgumentParser(description="Saliency Map Prediction")
    parser.add_argument(
        "--img_path",
        type=str,
        default="/media/data/WWZ/HZY/ml_hw1/009.jpg",
    )
    parser.add_argument("--condition", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument(
        "--heat_map_type",
        type=str,
        default="HOT",
        choices=["HOT", "Overlay"],
        help="Type of heatmap: HOT or Overlay",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = setup_model(device)

    pred_saliency, _ = saliency_map_prediction(
        args.img_path, args.condition, model, device
    )

    filename = os.path.splitext(os.path.basename(args.img_path))[0]
    hot_output_filename = os.path.join(args.output_path, f"{filename}_saliencymap.png")
    cv2.imwrite(
        hot_output_filename,
        (255 * pred_saliency / pred_saliency.max()).astype(np.uint8),
    )
    print(f"Saved HOT saliency map to {hot_output_filename}")


if __name__ == "__main__":
    main()
