import copy
import torch

import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.loss_function import SaliencyLoss
from utils.data_process_uni import SODDataset

from net.models.SUM import SUM
from net.configs.config_setting import setting_config

import matplotlib.pyplot as plt
import seaborn as sns


train_loss_history = []
val_loss_history = []

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_datasets = SODDataset(
    base_dir="../SOD/Saliency-TrainSet",
    transform=transform,
)
train_loader = DataLoader(train_datasets, batch_size=8, shuffle=True, num_workers=4)


val_dataset = SODDataset(
    base_dir="../SOD/Saliency-TestSet",
    transform=transform,
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)


def loss_compute(cc, kl, sim, mse):
    loss1 = -2 * cc + 10 * kl - sim
    loss2 = mse
    loss = loss1 + 5 * loss2
    return loss


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
        load_ckpt_path=model_cfg["load_ckpt_path"],
    )
    model.load_from()
    model.cuda()

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=35e-5, weight_decay=15e-3)

scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.15)
loss_fn = SaliencyLoss()
mse_loss = nn.MSELoss()

# Training and Validation Loop
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float("inf")
num_epochs = 100

# Early stopping setup
early_stop_counter = 0
early_stop_threshold = 5

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training Phase
    model.train()
    metrics = {"loss": [], "kl": [], "cc": [], "sim": []}

    for batch in tqdm(train_loader, desc="Training"):
        stimuli, smap, condition = (
            batch["image"].to(device),
            batch["saliency"].to(device),
            batch["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(stimuli, condition)

        # Compute losses
        kl = loss_fn(outputs, smap, loss_type="kldiv")
        cc = loss_fn(outputs, smap, loss_type="cc")
        sim = loss_fn(outputs, smap, loss_type="sim")

        loss = loss_compute(cc, kl, sim, mse_loss(outputs, smap))

        loss.backward()
        optimizer.step()

        metrics["loss"].append(loss.item())
        metrics["kl"].append(kl.item())
        metrics["cc"].append(cc.item())
        metrics["sim"].append(sim.item())

    scheduler.step()

    train_loss = np.mean(metrics["loss"])
    train_loss_history.append(train_loss)

    # 计算每个指标的平均值和标准差
    for metric in metrics.keys():
        metrics[metric] = (np.mean(metrics[metric]), np.std(metrics[metric]))

    print(
        "Train - "
        + ", ".join(
            [
                f"{metric}: {mean:.4f} ± {std:.4f}"
                for metric, (mean, std) in metrics.items()
            ]
        )
    )

    # Validation Phase
    model.eval()
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

            loss = loss_compute(cc, kl, sim, mse_loss(outputs, smap))

            val_metrics["loss"].append(loss.item())
            val_metrics["kl"].append(kl.item())
            val_metrics["cc"].append(cc.item())
            val_metrics["sim"].append(sim.item())

    val_loss = np.mean(val_metrics["loss"])
    val_loss_history.append(val_loss)

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
    print(f"Epoch {epoch+1}: Total Val Loss across all datasets: {total_val_loss:.4f}")

    # Check for best model
    if total_val_loss < best_loss:
        print(f"New best model found at epoch {epoch+1}!")
        best_loss = total_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "best_model.pth")
        early_stop_counter = 0  # Reset counter after improvement
    else:
        early_stop_counter += 1
        print(f"No improvement in Total Val Loss for {early_stop_counter} epoch(s).")

    # Early stopping check
    if early_stop_counter >= early_stop_threshold:
        print("Early stopping triggered.")
        break

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.3)

plt.figure(figsize=(12, 6))

plt.plot(
    range(1, len(train_loss_history) + 1),
    train_loss_history,
    label="Train Loss",
    color="blue",
    linestyle="-",
    marker="o",
    linewidth=2,
)
plt.plot(
    range(1, len(val_loss_history) + 1),
    val_loss_history,
    label="Validation Loss",
    color="red",
    linestyle="--",
    marker="s",
    linewidth=2,
)

# 添加标签和标题
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Training and Validation Loss", fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("loss_curve.pdf", dpi=600)
