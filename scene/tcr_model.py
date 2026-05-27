import os

import torch

from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration
from utils.tcr_utils import TCRNetwork


class TCRModel:
    """Temporal consistency residual head."""

    def __init__(self):
        self.tcr = TCRNetwork().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 0.1

    def step(self, x):
        return self.tcr(x)

    def train_setting(self, training_args):
        params = [
            {
                "params": list(self.tcr.parameters()),
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "tcr",
            }
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        self.tcr_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.tcr_lr_max_steps,
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "TCR/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.tcr.state_dict(), os.path.join(out_weights_path, "tcr.pth"))

    def _resolve_weights_path(self, model_path, iteration=-1):
        root = os.path.join(model_path, "TCR")
        loaded_iter = searchForMaxIteration(root) if iteration == -1 else iteration
        weights_path = os.path.join(root, "iteration_{}".format(loaded_iter), "tcr.pth")
        if os.path.exists(weights_path):
            return weights_path
        raise FileNotFoundError("No TCR checkpoint found under {}".format(model_path))

    def load_weights(self, model_path, iteration=-1):
        self.tcr.load_state_dict(torch.load(self._resolve_weights_path(model_path, iteration)))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "tcr":
                lr = self.tcr_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
        return None
