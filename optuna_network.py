import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.loss_function import SaliencyLoss
from utils.data_process_uni import SODDataset

from net.models.SUM import SUM
from net.configs.config_setting import setting_config

import optuna

transform = transforms.Compose(
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


sub_train_datasets = SubsetDataset(
    SODDataset(
        base_dir="/media/data/WWZ/HZY/ml_hw1/SOD/Saliency-TrainSet", transform=transform
    ),
    subset_ratio=0.4,
)

sub_val_dataset = SubsetDataset(
    SODDataset(
        base_dir="/media/data/WWZ/HZY/ml_hw1/SOD/Saliency-TestSet",
        transform=transform,
    ),
    subset_ratio=0.4,
)


def mean_std(test_list):
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance**0.5
    return mean, res


def objective(trial):
    log_file_path = "optuna_logs.txt"
    # Suggest values for the hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 9e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 9e-3)
    batch_size = 2 ** trial.suggest_int("batch_size", 1, 4)
    step_size = trial.suggest_int("step_size", 1, 6)
    gamma = trial.suggest_uniform("gamma", 0.05, 0.3)
    coef_kl = trial.suggest_float("coef_kl", 1, 20)
    coef_cc = trial.suggest_float("coef_cc", -5, 0)
    coef_sim = trial.suggest_float("coef_sim", -5, 0)
    coef_mse = trial.suggest_float("coef_mse", 0, 10)

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
        model.cuda(0)

    train_loader = DataLoader(
        sub_train_datasets, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        sub_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = SaliencyLoss()
    mse_loss = nn.MSELoss()

    # Initialize best loss for this trial
    best_loss = float("inf")

    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            stimuli, smap, condition = (
                batch["image"].to(device),
                batch["saliency"].to(device),
                batch["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(stimuli, condition)

            kl = loss_fn(outputs, smap, loss_type="kldiv")
            cc = loss_fn(outputs, smap, loss_type="cc")
            sim = loss_fn(outputs, smap, loss_type="sim")
            # nss = loss_fn(outputs, fmap, loss_type="nss")

            # loss1 = coef_cc * cc + coef_kl * kl + coef_sim * sim + coef_nss * nss
            loss1 = coef_cc * cc + coef_kl * kl + coef_sim * sim
            loss2 = mse_loss(outputs, smap)
            loss = loss1 + coef_mse * loss2

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * stimuli.size(0)

        scheduler.step()

        # Validation phase
        model.eval()
        val_kl = 0.0
        val_cc = 0.0
        val_sim = 0.0
        val_nss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                stimuli, smap, condition = (
                    batch["image"].to(device),
                    batch["saliency"].to(device),
                    batch["label"].to(device),
                )
                outputs = model(stimuli, condition)

                kl = loss_fn(outputs, smap, loss_type="kldiv")
                cc = loss_fn(outputs, smap, loss_type="cc")
                sim = loss_fn(outputs, smap, loss_type="sim")
                # nss = loss_fn(outputs, fmap, loss_type="nss")

                val_cc += cc.item() * stimuli.size(0)
                val_sim += sim.item() * stimuli.size(0)
                # val_nss += nss.item() * stimuli.size(0)
                val_kl += kl.item() * stimuli.size(0)

            combined_loss = val_kl - (val_cc + val_sim + val_nss)

            # Update best loss if the current validation KL is lower
            if combined_loss < best_loss:
                best_loss = combined_loss
    # After each trial, log the results
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Trial {trial.number}, Loss: {best_loss}\n")
        log_file.write(
            f"  Params: lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}, step_size: {step_size}, gamma: {gamma}, coef_kl: {coef_kl}, coef_cc: {coef_cc}, coef_sim: {coef_sim}, coef_mse: {coef_mse}\n"
        )

    return best_loss


# Create a study with specified storage, direction, and name
study = optuna.create_study(
    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
    study_name="quadratic-simple",
    direction="minimize",  # Specify the optimization direction here.
)

# Optimize the study
study.optimize(objective, n_trials=100)

# Logging the best trial information
with open("optimization_log_final.txt", "a") as log_file:
    log_file.write("Best trial:\n")
    log_file.write(f"  Value: {study.best_trial.value}\n")
    log_file.write("  Params: \n")
    for key, value in study.best_trial.params.items():
        log_file.write(f"    {key}: {value}\n")

print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
