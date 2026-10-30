import math

import pytorch_lightning as pl
import torch

MONITORS = ("Val/Loss", "Train/Loss")


def _loss_value(outputs):
    if isinstance(outputs, dict):
        outputs = outputs.get("loss")
    if outputs is None:
        return None
    if isinstance(outputs, torch.Tensor):
        if outputs.numel() != 1:
            return None
        return float(outputs.detach().float().cpu().item())
    try:
        return float(outputs)
    except (TypeError, ValueError):
        return None


class CurriculumFeedback(pl.Callback):
    def __init__(self, state, controller, monitor="Val/Loss"):
        if monitor not in MONITORS:
            raise ValueError(f"monitor must be one of {MONITORS}, got {monitor!r}")
        self.state = state
        self.controller = controller
        self.monitor = monitor
        self._loss_sum = 0.0
        self._loss_count = 0

    def on_train_start(self, trainer, pl_module):
        self.state.set(stage=self.controller.stage, active=self.controller.is_active(trainer.current_epoch))

    def on_train_epoch_start(self, trainer, pl_module):
        self._loss_sum = 0.0
        self._loss_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.monitor != "Train/Loss":
            return
        loss = _loss_value(outputs)
        if loss is not None and math.isfinite(loss):
            self._loss_sum += loss
            self._loss_count += 1

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        loss = None
        if trainer.is_global_zero:
            if self.monitor == "Val/Loss":
                value = trainer.callback_metrics.get("Val/Loss")
                loss = None if value is None else _loss_value(value)
            elif self._loss_count > 0:
                loss = self._loss_sum / self._loss_count
            decision = self.controller.update(epoch, loss)
        else:
            decision = (0, False)
        stage, active = trainer.strategy.broadcast(decision, src=0)
        self.state.set(stage=int(stage), active=bool(active))

        N, t_pca, p = self.state.params()
        pl_module.log_dict(
            {
                "SlideMix/stage": float(stage),
                "SlideMix/N": float(N),
                "SlideMix/T_PCA": float(t_pca),
                "SlideMix/P": float(p),
                "SlideMix/active": float(bool(active)),
            },
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
        if trainer.is_global_zero:
            loss_str = "n/a" if loss is None else f"{loss:.4f}"
            print(
                f"[SlideMix] epoch {epoch} {self.monitor}={loss_str} -> next epoch: stage={stage} "
                f"N={N} T_PCA={t_pca:.2f} P={p:.2f} active={bool(active)}"
            )

    def state_dict(self):
        return self.controller.state_dict()

    def load_state_dict(self, state_dict):
        self.controller.load_state_dict(state_dict)
