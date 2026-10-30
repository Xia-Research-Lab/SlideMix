import os
import math
import torch
import itertools
import numpy as np
import polars as pl
from pathlib import Path
from functools import partial
from torch.utils.data import Dataset
from safetensors.torch import safe_open
from pytorch_lightning import LightningDataModule
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, DistributedSampler


from .CONSTS import DATASETS
from .utils import (
    random_sampling,
    z_order_sampling,
    local_box_sampling,
    hilbert_sampling,
)

MAX_WORKERS = min(32, (os.cpu_count() or 1) * 5)


class SlideDataset(Dataset):
    def __init__(self, args, mode, slidemix=None):
        super().__init__()
        self.args = args
        self.mode = mode
        self.slidemix = slidemix if mode == "Train" else None
        self._load_df()
        self._filter_data()
        if self.slidemix is not None:
            self._report_slidemix_files()

    def __len__(self):
        return self.df.height

    def _load_df(self):
        task_names = list(self.args.tasks.keys())
        split_target_key = f"fold_information_5fold-{self.args.fold}"
        self.slide_id_key = DATASETS[
            "TCGA" if "TCGA" in self.args.dataset_name else self.args.dataset_name
        ]["slide_id_key"]
        csv_path = (
            Path(__file__).resolve().parents[1]
            / "dataset_configs"
            / self.args.dataset_name
            / "task_description.csv"
        )
        cols_needed = [self.slide_id_key, split_target_key, *task_names]
        df = pl.read_csv(csv_path, columns=cols_needed)

        task_cols = [pl.col(name).is_not_null() for name in task_names]
        df = df.filter(pl.col(split_target_key) == self.mode)
        if task_cols:
            df = df.filter(pl.any_horizontal(task_cols))

        self.df = df

    @staticmethod
    def _process_row(row_dict, data_root, slide_id_key):
        slide_id = row_dict[slide_id_key]
        slide_path = data_root / f"{slide_id}.safetensors"
        if slide_path.exists():
            new_row_data = row_dict.copy()
            new_row_data["data_path"] = slide_path
            return [new_row_data]
        return []

    def _filter_data(self):
        print(
            f"Initializing {self.args.dataset_name} {self.mode} split... Scanning files using {MAX_WORKERS} threads..."
        )
        worker_func = partial(
            SlideDataset._process_row,
            data_root=Path(self.args.data_path),
            slide_id_key=self.slide_id_key,
        )
        rows_to_process = self.df.to_dicts()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_list_of_lists = list(executor.map(worker_func, rows_to_process))

        processed_rows = list(itertools.chain.from_iterable(results_list_of_lists))
        self.df = pl.DataFrame(processed_rows)
        print(
            f"{self.mode} dataset initialized: {self.df.height} safetensors files found."
        )

    def _load_coords(self, data_path):
        with safe_open(data_path, framework="pt", device="cpu") as f:
            coords_yx = f.get_tensor("coords_yx")
        return coords_yx

    def _load_full_features(self, data_path):
        with safe_open(data_path, framework="pt", device="cpu") as f:
            image_features = f.get_tensor("features")
        return image_features

    def _load_roi(self, slide_id):
        roi_dir = self.slidemix.roi_dir if self.slidemix is not None else None
        if roi_dir is None:
            return None
        roi_path = Path(roi_dir) / f"{slide_id}.safetensors"
        if not roi_path.exists():
            return None
        with safe_open(roi_path, framework="pt", device="cpu") as f:
            return f.get_tensor("roi_mask").bool()

    def _load_vlm(self, slide_id):
        vlm_dir = self.slidemix.vlm_dir if self.slidemix is not None else None
        if vlm_dir is None:
            return {}
        vlm_path = Path(vlm_dir) / f"{slide_id}.safetensors"
        if not vlm_path.exists():
            return {}
        with safe_open(vlm_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            return {
                task: f.get_tensor(task).float()
                for task in self.args.tasks.keys()
                if task in keys
            }

    def _report_slidemix_files(self):
        cfg = self.slidemix
        slide_ids = [
            self.df.row(i, named=True)[self.slide_id_key] for i in range(len(self))
        ]

        def _count(folder):
            if folder is None:
                return f"0/{len(slide_ids)} (no dir)"
            found = sum(
                (Path(folder) / f"{sid}.safetensors").exists() for sid in slide_ids
            )
            return f"{found}/{len(slide_ids)}"

        fallback = ", linear fallback for missing" if cfg.label_method == "vlm" else ""
        print(
            f"SlideMix: ROI masks {_count(cfg.roi_dir)} (missing -> all tiles eligible), "
            f"VLM scores {_count(cfg.vlm_dir)} -> label {cfg.label_method}{fallback}"
        )

    def _load_features_by_indices(self, data_path, indices):
        with safe_open(data_path, framework="pt", device="cpu") as f:
            image_features = f.get_tensor("features")
            image_features = image_features[indices]
        return image_features

    def _sample_coords(self, coords_yx):
        if self.mode == "Train":
            if self.args.shuffle_tiles == "random":
                coords_yx, indices = random_sampling(coords_yx)

            elif self.args.shuffle_tiles == "z-order":
                coords_yx, indices = z_order_sampling(coords_yx)

            elif self.args.shuffle_tiles == "local-box":
                coords_yx, indices = local_box_sampling(coords_yx)

            elif self.args.shuffle_tiles == "hilbert":
                coords_yx, indices = hilbert_sampling(
                    coords_yx, sample_num=self.args.max_tiles
                )
            else:
                N = coords_yx.shape[0]
                indices = torch.arange(N)

        else:
            if self.args.shuffle_inf_tiles == "random":
                coords_yx, indices = random_sampling(coords_yx)

            elif self.args.shuffle_inf_tiles == "z-order":
                coords_yx, indices = z_order_sampling(coords_yx)

            elif self.args.shuffle_inf_tiles == "local-box":
                coords_yx, indices = local_box_sampling(coords_yx)

            elif self.args.shuffle_inf_tiles == "hilbert":
                coords_yx, indices = hilbert_sampling(
                    coords_yx, sample_num=self.args.max_inf_tiles
                )
            else:
                N = coords_yx.shape[0]
                indices = torch.arange(N)

        return coords_yx, indices

    def _truncate_or_get_indices(self, coords_yx, indices=None):
        N = coords_yx.shape[0]

        if self.mode == "Train":
            if self.args.max_tiles is not None and N >= self.args.max_tiles:
                coords_yx = coords_yx[: self.args.max_tiles]
                indices = indices[: self.args.max_tiles]
                pad_size = 0
            else:
                pad_size = self.args.max_tiles - N
                coords_yx = torch.cat(
                    (
                        coords_yx,
                        torch.zeros(
                            (pad_size, 2),
                            dtype=coords_yx.dtype,
                            device=coords_yx.device,
                        ),
                    ),
                    dim=0,
                )

        else:
            pad_size = 0
            if self.args.max_inf_tiles is not None and N >= self.args.max_inf_tiles:
                coords_yx = coords_yx[: self.args.max_inf_tiles]
                indices = indices[: self.args.max_inf_tiles]

        return coords_yx, indices, pad_size

    def _pad_features(self, image_features, pad_size):
        if pad_size > 0:
            D = image_features.shape[1]
            image_features = torch.cat(
                (
                    image_features,
                    torch.zeros(
                        (pad_size, D),
                        dtype=image_features.dtype,
                        device=image_features.device,
                    ),
                ),
                dim=0,
            )
        return image_features

    def _parse_labels(self, item_row):
        raw = {}
        for task, task_info in self.args.tasks.items():
            label = item_row.get(task, None)
            if label is None or (isinstance(label, float) and math.isnan(label)):
                label = None
            else:
                label = str(label)

            if task_info["type"] == "cls":
                raw[task] = task_info["labels"].get(label, None)

            elif task_info["type"] == "surv":
                if label:
                    time_str, event_str = label.split("|")
                    raw[task] = (float(time_str), float(event_str))
                else:
                    raw[task] = None

            else:
                raw[task] = float(label) if label else None

        return raw

    def _encode_labels(self, raw):
        task_labels = {}
        for task, task_info in self.args.tasks.items():
            value = raw[task]
            if task_info["type"] == "cls":
                if value is not None:
                    task_labels[task] = torch.tensor(value, dtype=torch.long)
                else:
                    task_labels[task] = torch.tensor(-1, dtype=torch.long)

            elif task_info["type"] == "surv":
                if value is not None:
                    task_labels[task] = torch.tensor(list(value), dtype=torch.float32)
                else:
                    task_labels[task] = torch.tensor([-1.0, -1.0], dtype=torch.float32)

            else:
                if value is not None:
                    task_labels[task] = torch.tensor(value, dtype=torch.float32)
                else:
                    task_labels[task] = torch.tensor(-1.0, dtype=torch.float32)

        return task_labels

    def __getitem__(self, idx):
        if self.slidemix is not None:
            return self._getitem_slidemix(idx)

        item_row = self.df.row(idx, named=True)
        slide_path = item_row["data_path"]

        coords_yx = self._load_coords(slide_path)
        coords_yx, indices = self._sample_coords(coords_yx)
        coords_yx, indices, pad_size = self._truncate_or_get_indices(coords_yx, indices)
        image_features = self._load_features_by_indices(slide_path, indices)
        image_features = self._pad_features(image_features, pad_size)

        task_labels = self._encode_labels(self._parse_labels(item_row))

        return {
            "image_features": image_features,
            "coords_yx": coords_yx,
            "task_labels": task_labels,
        }

    @staticmethod
    def _per_tile(tensor, n_tiles):
        if tensor is None or tensor.shape[0] != n_tiles:
            return None
        return tensor

    def _getitem_slidemix(self, idx):
        from .slidemix import (
            its_mix,
            soft_cls_target,
            soft_reg_target,
            vlm_evidence_weights,
        )

        cfg = self.slidemix
        item_row = self.df.row(idx, named=True)
        slide_path = item_row["data_path"]
        coords_ref = self._load_coords(slide_path)
        full_features = self._load_full_features(slide_path)
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))

        mix, partner_row, n_mov = None, None, 0
        if cfg.state.active and len(self) > 1 and rng.random() < cfg.prob:
            j = int(rng.integers(0, len(self) - 1))
            j = j + 1 if j >= idx else j
            partner_row = self.df.row(j, named=True)
            coords_mov = self._load_coords(partner_row["data_path"])
            features_mov = self._load_full_features(partner_row["data_path"])
            n_mov = coords_mov.shape[0]
            roi_ref = self._per_tile(
                self._load_roi(item_row[self.slide_id_key]), coords_ref.shape[0]
            )
            roi_mov = self._per_tile(
                self._load_roi(partner_row[self.slide_id_key]), n_mov
            )
            N, T_PCA, P = cfg.state.params()
            mix = its_mix(
                full_features,
                coords_ref,
                features_mov,
                coords_mov,
                N=N,
                P=P,
                T_PCA=T_PCA,
                roi_r=roi_ref,
                roi_m=roi_mov,
                pca_stats=cfg.pca_stats,
                rng=rng,
            )
            full_features = mix.features

        coords_yx, indices = self._sample_coords(coords_ref)
        coords_yx, indices, pad_size = self._truncate_or_get_indices(coords_yx, indices)
        image_features = self._pad_features(full_features[indices], pad_size)

        f = 1.0
        if mix is not None and len(indices) > 0:
            f = 1.0 - mix.source[indices].float().mean().item()
        mixed = f < 1.0

        raw_ref = self._parse_labels(item_row)
        raw_mov = self._parse_labels(partner_row) if mixed else None
        method = cfg.label_method if mixed else "linear"
        label_seed = int(rng.integers(0, 2**31 - 1))
        vlm_ref, vlm_mov = {}, {}
        if mixed and method == "vlm":
            vlm_ref = self._load_vlm(item_row[self.slide_id_key])
            vlm_mov = self._load_vlm(partner_row[self.slide_id_key])

        task_labels = {}
        for task, task_info in self.args.tasks.items():
            label_rng = np.random.default_rng(label_seed)
            y_ref = raw_ref[task]
            y_mov = raw_mov[task] if mixed else None

            if task_info["type"] == "cls":
                w_ref, w_mov = None, None
                if mixed and method == "vlm":
                    w_ref, w_mov = vlm_evidence_weights(
                        self._per_tile(vlm_ref.get(task), coords_ref.shape[0]),
                        self._per_tile(vlm_mov.get(task), n_mov),
                        mix.source[indices],
                        indices,
                        mix.moving_index[indices],
                        y_ref,
                        y_mov,
                    )
                task_labels[task] = soft_cls_target(
                    y_ref,
                    y_mov,
                    len(task_info["labels"]),
                    f,
                    method,
                    label_rng,
                    w_ref,
                    w_mov,
                )

            elif task_info["type"] == "surv":
                task_labels[task] = torch.tensor(
                    list(y_ref) if y_ref is not None else [-1.0, -1.0],
                    dtype=torch.float32,
                )

            else:
                value = soft_reg_target(y_ref, y_mov, f, method, label_rng)
                task_labels[task] = torch.tensor(value, dtype=torch.float32)

        return {
            "image_features": image_features,
            "coords_yx": coords_yx,
            "task_labels": task_labels,
        }


class data_module(LightningDataModule):
    def __init__(self, args, slidemix=None):
        super().__init__()
        self.args = args
        self.slidemix = slidemix

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = SlideDataset(self.args, "Train", slidemix=self.slidemix)
            self.val_set = SlideDataset(self.args, "Val")
        elif stage == "test":
            self.test_set = SlideDataset(self.args, "Test")

    def _get_sampler(self, dataset, shuffle):
        if self.trainer and self.trainer._accelerator_connector.is_distributed:
            return DistributedSampler(dataset, shuffle=shuffle, seed=self.args.seed)
        return None

    def _create_dataloader(self, dataset, shuffle=False):
        sampler = self._get_sampler(dataset, shuffle)
        return DataLoader(
            dataset=dataset,
            pin_memory=True,
            sampler=sampler,
            drop_last=shuffle,
            num_workers=self.args.num_workers
            if self.args.num_workers >= 0
            else os.cpu_count() // len(self.args.devices),
            shuffle=(sampler is None and shuffle),
            persistent_workers=self.args.num_workers > 0,
            batch_size=self.args.batch_size if shuffle else 1,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_set, shuffle=False)


def prepare_pca_stats(args, path, k=16, num_slides=256, seed=0, source=None):
    from .slidemix import PCAStats, fit_pca_stats

    path = Path(path)
    if path.exists():
        stats = PCAStats.load(path)
        print(
            f"SlideMix PCA stats loaded from {path}: k={stats.components.shape[1]}, "
            f"d_min={stats.d_min:.4g}, d_max={stats.d_max:.4g}"
        )
        return stats

    if source is None:
        source = SlideDataset(args, "Train")
    if isinstance(source, SlideDataset):
        slide_paths = [
            source.df.row(i, named=True)["data_path"] for i in range(len(source))
        ]
    else:
        slide_paths = list(source)
    if not slide_paths:
        raise ValueError("prepare_pca_stats: no training slides to fit PCA stats on")

    rng = np.random.default_rng(seed)
    take = min(int(num_slides), len(slide_paths))
    chosen = [slide_paths[i] for i in rng.choice(len(slide_paths), size=take, replace=False)]

    def _features():
        for slide_path in chosen:
            with safe_open(slide_path, framework="pt", device="cpu") as f:
                yield f.get_tensor("features").float()

    stats = fit_pca_stats(_features(), k=k, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    stats.save(tmp_path)
    os.replace(tmp_path, path)
    print(
        f"SlideMix PCA stats fitted on {take}/{len(slide_paths)} Train slides "
        f"(k={stats.components.shape[1]}, d_min={stats.d_min:.4g}, d_max={stats.d_max:.4g}) -> {path}"
    )
    return stats
