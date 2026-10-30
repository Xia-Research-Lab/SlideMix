import sys
import torch
import argparse
import warnings
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))

from utils.model import WSI_Model
from utils.dataset import data_module
from utils.utils import initialize_experiment


torch.set_float32_matmul_precision("high")
warnings.filterwarnings(
    "ignore", message=".*Metric `SpearmanCorrcoef` will save all targets.*"
)
warnings.filterwarnings(
    "ignore", message=".*Precision bf16-mixed is not supported by the model summary.*"
)


def get_args_parser():
    parser = argparse.ArgumentParser(description="WSI MIL Pipeline")

    group_env = parser.add_argument_group("Environment Settings")
    group_env.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    group_env.add_argument(
        "--devices",
        type=str,
        default="0",
        help="GPU IDs: '0', '0%1', or 'auto' for all available GPUs (default: '0')",
    )
    group_env.add_argument(
        "--runs_path",
        type=str,
        default=str(ROOT_PATH / "runs"),
        help="Path to save logs/results",
    )
    group_env.add_argument(
        "--runs_name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    group_env.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Running mode (default: 'train')",
    )
    group_env.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Optional path to write test results JSON",
    )

    group_data = parser.add_argument_group("Data Settings")
    group_data.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Dataset root path",
    )
    group_data.add_argument(
        "--dataset_name",
        type=str,
        default="CAMELYON16",
        help="Dataset name (e.g., CAMELYON16, PANDA)",
    )
    group_data.add_argument(
        "--embedding_name",
        type=str,
        default="GigaPath",
        help="Backbone used for feature extraction",
    )
    group_data.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Specific tasks (split by %), default: run all in config",
    )
    group_data.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Specific fold, default: 1",
    )
    group_data.add_argument(
        "--num_workers", type=int, default=32, help="Dataloader workers"
    )
    group_data.add_argument(
        "--max_tiles",
        type=int,
        default=2048,
        help="Max tiles per slide duirng training (sampling/padding limit)",
    )
    group_data.add_argument(
        "--max_inf_tiles",
        type=int,
        default=None,
        help="Max tiles per slide during inference",
    )
    group_data.add_argument(
        "--shuffle_tiles",
        type=str,
        default="random",
        choices=["None", "random", "z-order", "local-box", "hilbert"],
        help="Tile shuffling strategy",
    )
    group_data.add_argument(
        "--shuffle_inf_tiles",
        type=str,
        default="None",
        choices=["None", "random", "z-order", "local-box", "hilbert"],
        help="Tile shuffling strategy",
    )

    group_model = parser.add_argument_group("Model Settings")
    group_model.add_argument(
        "--model_name", type=str, default="ABMIL", help="MIL Aggregator name"
    )
    group_model.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to load trained checkpoint for continue training, 'None' for init from scratch",
    )
    group_model.add_argument(
        "--test_ckpt",
        type=str,
        default=None,
        help="Path to load checkpoint for testing, 'None' for default best.ckpt in runs_path",
    )

    group_train = parser.add_argument_group("Training Settings")
    group_train.add_argument("--batch_size", type=int, default=4, help="Batch size")
    group_train.add_argument("--epochs", type=int, default=100, help="Total epochs")
    group_train.add_argument(
        "--val_interval",
        type=float,
        default=1.0,
        help="Validation interval, 1.0 for epoch, 1 for step",
    )
    group_train.add_argument(
        "--early_stop_epoch", type=int, default=10, help="Patience for early stopping"
    )
    group_train.add_argument(
        "--lr", type=float, default=1e-04, help="Learning rate (default: 1e-4)"
    )
    group_train.add_argument(
        "--lrf",
        type=float,
        default=0.1,
        help="Cosine annealing end learning rate ratio",
    )
    group_train.add_argument(
        "--disable_pbar", action="store_true", help="Disable tqdm progress bar"
    )
    group_train.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Training precision",
    )

    group_slidemix = parser.add_argument_group("SlideMix Settings")
    group_slidemix.add_argument(
        "--slidemix",
        action="store_true",
        help="Enable SlideMix augmentation on the training split (default: False)",
    )
    group_slidemix.add_argument(
        "--slidemix_prob",
        type=float,
        default=0.5,
        help="Probability that a training slide is mixed with a random partner",
    )
    group_slidemix.add_argument(
        "--slidemix_N",
        type=str,
        default="16,8,4,2,1",
        help="Curriculum schedule of the block granularity N (comma separated)",
    )
    group_slidemix.add_argument(
        "--slidemix_TPCA",
        type=str,
        default="1.0,0.8,0.6,0.4,0.2",
        help="Curriculum schedule of the PCA similarity threshold T_PCA (comma separated)",
    )
    group_slidemix.add_argument(
        "--slidemix_P",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5",
        help="Curriculum schedule of the shuffling ratio P (comma separated)",
    )
    group_slidemix.add_argument(
        "--slidemix_disable",
        type=str,
        default="",
        help="Schedulers pinned at their stage-0 value, comma subset of N,TPCA,P",
    )
    group_slidemix.add_argument(
        "--slidemix_strategy",
        type=str,
        default="hold",
        choices=["hold", "back", "fixed"],
        help="Curriculum strategy when the loss misses the threshold",
    )
    group_slidemix.add_argument(
        "--slidemix_warmup_epochs",
        type=int,
        default=20,
        help="Epochs trained without mixing before the curriculum starts",
    )
    group_slidemix.add_argument(
        "--slidemix_rel_drop",
        type=float,
        default=0.05,
        help="Relative loss drop needed to advance a stage (T_loss = ref_loss * (1 - rel_drop))",
    )
    group_slidemix.add_argument(
        "--slidemix_loss_threshold",
        type=float,
        default=None,
        help="Absolute loss threshold T_loss (overrides --slidemix_rel_drop)",
    )
    group_slidemix.add_argument(
        "--slidemix_min_epochs",
        type=int,
        default=1,
        help="Minimum epochs per curriculum stage before a decision",
    )
    group_slidemix.add_argument(
        "--slidemix_monitor",
        type=str,
        default="Val/Loss",
        choices=["Val/Loss", "Train/Loss"],
        help="Loss driving the curriculum",
    )
    group_slidemix.add_argument(
        "--slidemix_label",
        type=str,
        default="vlm",
        choices=["linear", "random", "vlm"],
        help="Soft label method for mixed slides (vlm falls back to linear without scores)",
    )
    group_slidemix.add_argument(
        "--slidemix_roi_dir",
        type=str,
        default=None,
        help="Folder of <slide_id>.safetensors ROI masks (default: every tile eligible)",
    )
    group_slidemix.add_argument(
        "--slidemix_vlm_dir",
        type=str,
        default=None,
        help="Folder of <slide_id>.safetensors per-tile VLM class scores",
    )
    group_slidemix.add_argument(
        "--slidemix_pca_k", type=int, default=16, help="PCA components for T_PCA"
    )
    group_slidemix.add_argument(
        "--slidemix_pca_slides",
        type=int,
        default=256,
        help="Train slides sampled to fit the PCA stats",
    )
    group_slidemix.add_argument(
        "--slidemix_pca_path",
        type=str,
        default=None,
        help="PCA stats cache (default: <runs_path>/slidemix_pca.pt)",
    )

    return parser.parse_args()


def build_slidemix(args):
    from utils.dataset import prepare_pca_stats
    from utils.slidemix import (
        CurriculumController,
        CurriculumSchedules,
        CurriculumState,
        SlideMixConfig,
    )
    from utils.slidemix_callback import CurriculumFeedback

    schedules = CurriculumSchedules(
        N=args.slidemix_N,
        T_PCA=args.slidemix_TPCA,
        P=args.slidemix_P,
        disabled=args.slidemix_disable,
    )
    state = CurriculumState(
        schedules, stage=0, active=args.slidemix_warmup_epochs == 0
    )
    controller = CurriculumController(
        num_stages=schedules.num_stages,
        strategy=args.slidemix_strategy,
        warmup_epochs=args.slidemix_warmup_epochs,
        rel_drop=args.slidemix_rel_drop,
        abs_threshold=args.slidemix_loss_threshold,
        min_epochs_per_stage=args.slidemix_min_epochs,
    )

    reachable_tpca = (
        schedules.T_PCA[:1] if "TPCA" in schedules.disabled else schedules.T_PCA
    )
    pca_stats = None
    if min(reachable_tpca) < 1.0:
        pca_path = (
            Path(args.slidemix_pca_path)
            if args.slidemix_pca_path
            else Path(args.runs_path) / "slidemix_pca.pt"
        )
        pca_stats = prepare_pca_stats(
            args,
            pca_path,
            k=args.slidemix_pca_k,
            num_slides=args.slidemix_pca_slides,
            seed=args.seed,
        )

    cfg = SlideMixConfig(
        state=state,
        prob=args.slidemix_prob,
        label_method=args.slidemix_label,
        pca_stats=pca_stats,
        roi_dir=args.slidemix_roi_dir,
        vlm_dir=args.slidemix_vlm_dir,
    )
    callback = CurriculumFeedback(state, controller, monitor=args.slidemix_monitor)
    print(f"SlideMix enabled: {state}, strategy={args.slidemix_strategy}")
    return cfg, callback


if __name__ == "__main__":
    args = get_args_parser()
    seed_everything(args.seed, workers=True)
    (
        args.data_path,
        args.input_dim,
        args.tasks,
        args.runs_path,
        args.runs_name,
        args.devices,
    ) = initialize_experiment(args)

    if args.mode == "train":
        tb_logger = TensorBoardLogger(
            version="tb",
            name=args.runs_name,
            default_hp_metric=False,
            save_dir=args.runs_path.parent,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.runs_path / "checkpoints",
            filename="best",
            monitor="Val/Loss",
            mode="min",
            save_top_k=1,
        )
        early_stop_callback = EarlyStopping(
            monitor="Val/Loss",
            min_delta=0.00001,
            patience=args.early_stop_epoch,
            verbose=True,
            mode="min",
        )
        callbacks = [
            checkpoint_callback,
            early_stop_callback,
        ]
    else:
        tb_logger = False
        callbacks = None

    slidemix_cfg = None
    if args.mode == "train" and args.slidemix:
        slidemix_cfg, slidemix_callback = build_slidemix(args)
        callbacks.append(slidemix_callback)

    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        enable_model_summary=True,
        val_check_interval=args.val_interval,
        enable_progress_bar=not args.disable_pbar,
        strategy="ddp" if len(args.devices) != 1 else "auto",
    )

    if slidemix_cfg is not None:
        dm = data_module(args, slidemix=slidemix_cfg)
    else:
        dm = data_module(args)

    if args.mode == "train":
        model = WSI_Model(args)
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt)
        model = WSI_Model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path, args=args, weights_only=False
        )
        trainer.test(model, datamodule=dm)

    else:
        ckpt_path = (
            args.test_ckpt
            if args.test_ckpt
            else args.runs_path / "checkpoints" / "best.ckpt"
        )
        print(f"Loading checkpoint from {ckpt_path}")
        model = WSI_Model.load_from_checkpoint(ckpt_path, args=args, weights_only=False)
        trainer.test(model, datamodule=dm)
