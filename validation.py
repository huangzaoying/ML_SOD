import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import transforms

from tqdm import tqdm

from utils.loss_function import SaliencyLoss
from utils.data_process_uni import SODDataset

from net.models.SUM import SUM
from net.configs.config_setting import setting_config


val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class SubsetDataset(Dataset):
    def __init__(self, base_dataset, subset_ratio=0.20):
        self.base_dataset = base_dataset
        total_count = len(self.base_dataset)
        subset_count = int(total_count * subset_ratio)
        self.indices = torch.randperm(total_count)[:subset_count].tolist()

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


# Instantiate a ValDataset for each validation dataset
val_dataset = SODDataset(
    base_dir="/media/data/WWZ/HZY/ml_hw1/SOD/Saliency-TestSet",
    transform=val_transform,
)

# Load validation datasets with subset
sub_val_dataset = SubsetDataset(val_dataset, subset_ratio=1)

# Create a DataLoader for each ValDataset
val_loader = DataLoader(sub_val_dataset, batch_size=16, shuffle=False, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    model.load_from()
    model.cuda()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

# Load the pre-trained model weights
model.load_state_dict(
    torch.load(
        "./best_model.pth",
        map_location=device,
    )
)


# Function for performing validation inference
def perform_validation_inference(val_loaders, model, device):
    loss_fn = SaliencyLoss()
    model.eval()  # Set model to evaluation mode
    val_metrics = {"loss": [], "kl": [], "cc": [], "sim": []}

    for batch in tqdm(val_loader, desc="Validating"):
        stimuli, smap, condition = (
            batch["image"].to(device),
            batch["saliency"].to(device),
            batch["label"].to(device),
        )

        with torch.no_grad():
            outputs = model(stimuli, condition)

            # Compute losses
            kl = loss_fn(outputs, smap, loss_type="kldiv")
            cc = loss_fn(outputs, smap, loss_type="cc")
            sim = loss_fn(outputs, smap, loss_type="sim")

            loss = -2 * cc + 10 * kl - 1 * sim

            # Accumulate raw metric values for validation
            val_metrics["loss"].append(loss.item())
            val_metrics["kl"].append(kl.item())
            val_metrics["cc"].append(cc.item())
            val_metrics["sim"].append(sim.item())

    # Calculate mean and std dev for each validation metric
    # 计算每个指标的平均值和标准差
    for metric in val_metrics.keys():
        val_metrics[metric] = (
            np.mean(val_metrics[metric]),
            np.std(val_metrics[metric]),
        )

    metrics_str = ", ".join(
        [
            f"{metric}: {mean:.4f} ± {std:.4f}"
            for metric, (mean, std) in val_metrics.items()
        ]
    )
    print(f"Val Metrics: {metrics_str}")

    total_val_loss = np.sum(val_metrics["kl"])
    print(f"Total Val Loss across all datasets: {total_val_loss:.4f}")


# Perform validation inference
perform_validation_inference(val_loader, model, device)
