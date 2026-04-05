from collections import defaultdict

import torch

from src.models.bpr_recommender import PyTorchBPR, BPRMFModel


def load_bpr_model(model_path: str, device: str = "cpu") -> PyTorchBPR:
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint["config"]

    wrapper = PyTorchBPR(
        factors=config["factors"],
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        reg=config["reg"],
        positive_threshold=config["positive_threshold"],
        top_k=config["top_k"],
        patience=config["patience"],
        neg_alpha=config["neg_alpha"],
        seed=config["seed"],
        verbose=False,
    )

    wrapper.device = torch.device(device)
    wrapper.user_map = checkpoint["user_map"]
    wrapper.item_map = checkpoint["item_map"]
    wrapper.reverse_user_map = checkpoint["reverse_user_map"]
    wrapper.reverse_item_map = checkpoint["reverse_item_map"]
    wrapper.num_users = checkpoint["num_users"]
    wrapper.num_items = checkpoint["num_items"]
    wrapper.best_epoch = checkpoint.get("best_epoch", 0)
    wrapper.best_val_map = checkpoint.get("best_val_map", -1)

    wrapper.all_train_seen = defaultdict(set)
    for u_idx, item_set in checkpoint["all_train_seen"].items():
        wrapper.all_train_seen[int(u_idx)] = set(item_set)

    wrapper.model = BPRMFModel(
        num_users=wrapper.num_users,
        num_items=wrapper.num_items,
        factors=wrapper.factors,
    ).to(wrapper.device)

    wrapper.model.load_state_dict(checkpoint["model_state_dict"])
    wrapper.model.eval()

    return wrapper