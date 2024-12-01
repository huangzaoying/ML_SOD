import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import transforms

from tqdm import tqdm

from utils.loss_function import SaliencyLoss
from utils.data_process_uni import SODCategoryDataset, get_datasets_by_category

from net.models.SUM import SUM
from net.configs.config_setting import setting_config
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_all_metrics(val_metrics):
    categories = list(val_metrics.keys())
    metrics = ["kl", "cc", "sim"]

    # 设置seaborn的主题样式
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # 设定颜色，分别对应四个指标
    colors = ["blue", "orange", "green", "red"]

    # 创建一个大的图，4个子图
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    axes = axes.flatten()  # 展开为一维数组方便使用

    for i, metric in enumerate(metrics):
        # 提取当前指标的均值和标准差
        means = [val_metrics[cat][metric][0] for cat in categories]
        stds = [val_metrics[cat][metric][1] for cat in categories]

        # 使用散点图和误差条绘制
        axes[i].scatter(
            categories, means, color=colors[i], label=f"{metric.upper()} Mean", zorder=5
        )
        axes[i].errorbar(
            categories,
            means,
            yerr=stds,
            fmt="o",
            color=colors[i],
            capsize=5,
            label=f"{metric.upper()} ± Std",
            zorder=3,
        )

        # 设置标签和标题
        if i == 1:  # 只有底部的图才显示x轴标签
            axes[i].set_xlabel("Categories", fontsize=12)

        axes[i].set_ylabel(f"{metric.upper()} Value", fontsize=12)
        axes[i].set_title(f"{metric.upper()} Across Categories", fontsize=14)

        # 控制横坐标标签，避免重叠，调整字体大小并垂直显示
        axes[i].tick_params(axis="x", rotation=90, labelsize=10)

        # 增加网格线
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

        axes[i].legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(
        "validation_all_metrics.pdf",
        dpi=600,
        bbox_inches="tight",
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


val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Instantiate a ValDataset for each validation dataset
val_datasets = get_datasets_by_category(
    base_dir="../SOD/Saliency-TestSet",
    transform=val_transform,
)

sub_val_datasets = {
    category: SubsetDataset(val_dataset, subset_ratio=1)
    for category, val_dataset in val_datasets.items()
}

val_loaders = {
    category: DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    for category, val_dataset in sub_val_datasets.items()
}

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
model.load_state_dict(torch.load("./best_model.pth", map_location=device))


# Function for performing validation inference
def perform_validation_inference(val_loaders, model, device):
    loss_fn = SaliencyLoss()
    model.eval()  # Set model to evaluation mode
    val_metrics = {
        name: {"loss": [], "kl": [], "cc": [], "sim": []} for name in val_loaders.keys()
    }
    for name, val_loader in val_loaders.items():
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
                val_metrics[name]["loss"].append(loss.item())
                val_metrics[name]["kl"].append(kl.item())
                val_metrics[name]["cc"].append(cc.item())
                val_metrics[name]["sim"].append(sim.item())

        # Calculate mean and std dev for each validation metric
        for metric in val_metrics[name].keys():
            val_metrics[name][metric] = (
                np.mean(val_metrics[name][metric]),
                np.std(val_metrics[name][metric]),
            )

        # Print val metrics
        metrics_str = ", ".join(
            [
                f"{metric}: {mean:.4f} ± {std:.4f}"
                for metric, (mean, std) in val_metrics[name].items()
            ]
        )
        print(f"{name} - Val Metrics: {metrics_str}")

    # total_val_loss = np.sum(val_metrics["kl"])
    total_val_loss = sum(
        [np.sum(val_metrics[name]["kl"]) for name in val_loaders.keys()]
    )
    # 在完成验证后调用该函数
    visualize_all_metrics(val_metrics)
    print(f"Total Val Loss across all datasets: {total_val_loss:.4f}")


# Perform validation inference
perform_validation_inference(val_loaders, model, device)
