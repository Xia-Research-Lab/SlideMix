import yaml
import torch
import numpy as np
import torch.nn as nn
from numba import njit
from pathlib import Path
from torchmetrics import Metric
from sklearn.neighbors import BallTree
from sksurv.metrics import concordance_index_censored

from .CONSTS import EMBEDDING_MODELS, DATASETS_ORGAN


class CoxPHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        times, events = targets[:, 0], targets[:, 1]
        sorted_idx = torch.argsort(times, descending=True)
        preds = preds[sorted_idx]
        events = events[sorted_idx]

        gamma = preds.max()
        exp_logits = torch.exp(preds - gamma)
        log_risk_set = torch.log(torch.cumsum(exp_logits, dim=0)) + gamma
        nll = -(preds - log_risk_set)
        masked_nll = nll * events
        num_events = events.sum()
        if num_events == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        return masked_nll.sum() / num_events


class SksurvCIndex(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("times", default=[], dist_reduce_fx="cat")
        self.add_state("events", default=[], dist_reduce_fx="cat")

    def update(self, preds, times, events):
        self.preds.append(preds.reshape(-1))
        self.times.append(times.reshape(-1))
        self.events.append(events.reshape(-1))

    def compute(self):
        preds = torch.cat(self.preds).detach().cpu().float().numpy()
        times = torch.cat(self.times).detach().cpu().float().numpy()
        events = torch.cat(self.events).detach().cpu().float().numpy()

        valid_mask = ~(np.isnan(preds) | np.isnan(times) | np.isnan(events))
        preds = preds[valid_mask]
        times = times[valid_mask]
        events = events[valid_mask]

        if len(preds) == 0:
            return torch.tensor(0.0)

        event_indicator = events.astype(bool)
        c_index = concordance_index_censored(event_indicator, times, preds)[0]
        return torch.tensor(c_index)


def initialize_experiment(args):
    embedding_folder = EMBEDDING_MODELS[args.embedding_name]["folder_name"]
    input_dim = EMBEDDING_MODELS[args.embedding_name]["dim"]
    data_path = Path(args.data_path) / args.dataset_name / embedding_folder

    task_config_path = (
        Path(__file__).resolve().parents[1]
        / "dataset_configs"
        / args.dataset_name
        / "task_configs.yaml"
    )
    with open(task_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    task_types = config["all_task_dict"]
    label_dict = config["label_dict"]

    if args.tasks is None:
        tasks_to_run = config["tasks_to_run"]
    elif isinstance(args.tasks, str):
        if "%" in args.tasks:
            tasks_to_run = args.tasks.split("%")
        else:
            tasks_to_run = [args.tasks]
    else:
        raise NotImplementedError(
            f"Unsupported type for tasks_to_run: {type(args.tasks)}"
        )

    def get_loss_func(task_type):
        if task_type == "cls":
            return nn.CrossEntropyLoss()
        elif task_type == "reg":
            return nn.SmoothL1Loss(beta=1.0)
        elif task_type == "surv":
            return CoxPHLoss()
        else:
            raise NotImplementedError(f"Unsupported task type: {task_type}")

    tasks = {}
    for task in tasks_to_run:
        tasks[task] = {
            "type": task_types[task],
            "labels": label_dict.get(task, None),
            "organ": DATASETS_ORGAN[args.dataset_name],
            "loss_func": get_loss_func(task_types[task]),
            "loss_weight": 1,
        }

    if args.runs_name:
        runs_name = args.runs_name
    else:
        tasks_str = "_".join(tasks.keys())
        runs_name = f"{args.embedding_name}_{tasks_str}_{args.lr:.0e}_B{args.batch_size}_T{args.max_tiles}_{args.shuffle_tiles}"
        if getattr(args, "slidemix", False):
            runs_name += "_slidemix"

    runs_path = (
        Path(args.runs_path)
        / args.dataset_name
        / args.model_name
        / f"Fold_{args.fold}"
        / runs_name
    )

    if args.mode == "train":
        runs_path.mkdir(parents=True, exist_ok=True)

    if args.devices != "auto":
        devices = [int(num_str) for num_str in args.devices.split("%")]
    else:
        devices = args.devices

    return [data_path, input_dim, tasks, runs_path, runs_name, devices]


def random_sampling(coords_yx):
    indices = torch.randperm(len(coords_yx))
    return coords_yx[indices], indices


def z_order_sampling(coords_yx):
    unique_x_coords = sorted(list(set(int(x.item()) for x in coords_yx[:, 0])))
    unique_y_coords = sorted(list(set(int(y.item()) for y in coords_yx[:, 1])))
    if not unique_x_coords or not unique_y_coords:
        return coords_yx, torch.arange(coords_yx.shape[0])

    H_dense = len(unique_y_coords)
    W_dense = len(unique_x_coords)
    min_hw = min(H_dense, W_dense)
    scan_level_size = 2 ** (min_hw.bit_length() - 1)

    x_map = {x_val: i for i, x_val in enumerate(unique_x_coords)}
    y_map = {y_val: i for i, y_val in enumerate(unique_y_coords)}

    full_grid = -torch.ones((H_dense, W_dense), dtype=torch.long)
    for idx, (x_sparse, y_sparse) in enumerate(coords_yx):
        xi_dense = x_map.get(int(x_sparse.item()))
        yi_dense = y_map.get(int(y_sparse.item()))

        if xi_dense is not None and yi_dense is not None:
            full_grid[yi_dense, xi_dense] = idx

    def generate_z_order_path(size: int) -> list[tuple[int, int]]:
        if not (size > 0 and (size & (size - 1) == 0)):
            raise ValueError(f"Size must be a power of 2, got {size}")
        path = []
        stack = [(0, 0, size)]
        while stack:
            x, y, s = stack.pop()
            if s == 1:
                path.append((x, y))
                continue
            s_half = s // 2
            stack.append((x + s_half, y + s_half, s_half))
            stack.append((x, y + s_half, s_half))
            stack.append((x + s_half, y, s_half))
            stack.append((x, y, s_half))
        return path

    try:
        z_path_pattern = generate_z_order_path(scan_level_size)

    except ValueError as e:
        print(f"Error generating Z-path: {e}")
        print(
            "Defaulting to original order or consider providing a valid power-of-2 scan_level_size."
        )
        return coords_yx, torch.arange(coords_yx.shape[0])

    reordered_original_indices = []
    for block_y_start_dense in range(0, H_dense, scan_level_size):
        for block_x_start_dense in range(0, W_dense, scan_level_size):
            for dx_in_tile, dy_in_tile in z_path_pattern:
                current_dense_x = block_x_start_dense + dx_in_tile
                current_dense_y = block_y_start_dense + dy_in_tile
                if 0 <= current_dense_x < W_dense and 0 <= current_dense_y < H_dense:
                    original_idx = full_grid[current_dense_y, current_dense_x]
                    if original_idx != -1:
                        reordered_original_indices.append(original_idx.item())

    reordered_indices_tensor = torch.tensor(
        reordered_original_indices, dtype=torch.long, device=coords_yx.device
    )
    return coords_yx[reordered_indices_tensor], reordered_indices_tensor


def local_box_sampling(
    coords_yx,
    bag_num=60,
    bag_size=64,
    neighbourhood_radius=4,
    tile_pixel_size=224,
    max_ittr=5,
):
    N = coords_yx.shape[0]
    if N < bag_size:
        return coords_yx, torch.arange(N)

    coords_np = coords_yx.numpy()
    tree = BallTree(coords_np, metric="manhattan")
    shuffled_indices = np.random.permutation(N)
    max_distance = neighbourhood_radius * tile_pixel_size

    attempts = 0
    bag_valid_flag = 0
    sampled_indices = []

    for center_idx in shuffled_indices:
        center_coord = coords_np[center_idx]
        query_result = tree.query_radius(center_coord.reshape(1, -1), r=max_distance)
        actual_indices = query_result[0]
        if len(actual_indices) < bag_size:
            continue

        selected = np.random.choice(actual_indices, size=bag_size, replace=False)
        sampled_indices.extend(selected.tolist())
        bag_valid_flag += 1

        if bag_valid_flag == bag_num:
            break

        attempts += 1
        if attempts > max_ittr * N:
            break

    if bag_valid_flag == 0:
        indices = np.random.choice(N, size=min(N, bag_num * bag_size), replace=True)
        indices_tensor = torch.from_numpy(indices.astype(np.int64))
        return coords_yx[indices_tensor], indices_tensor

    sampled_indices = np.array(sampled_indices, dtype=np.int64)
    indices_tensor = torch.from_numpy(sampled_indices)
    sampled_coords = coords_yx[indices_tensor]

    return sampled_coords, indices_tensor

@njit(fastmath=True)
def _xy2d_numba(n, x, y):
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s //= 2
    return d


@njit(fastmath=True)
def _compute_hilbert_distances(x_dense, y_dense, n_side):
    n = x_dense.shape[0]
    dists = np.empty(n, dtype=np.int64)
    for i in range(n):
        dists[i] = _xy2d_numba(n_side, x_dense[i], y_dense[i])
    return dists


def hilbert_sampling(coords_yx, sample_num=None):
    device = coords_yx.device

    x_raw = coords_yx[:, 0].long()
    y_raw = coords_yx[:, 1].long()

    unique_x, x_dense = torch.unique(x_raw, sorted=True, return_inverse=True)
    unique_y, y_dense = torch.unique(y_raw, sorted=True, return_inverse=True)

    if unique_x.numel() == 0 or unique_y.numel() == 0:
        return coords_yx, torch.arange(coords_yx.shape[0], device=device)

    max_dim = max(unique_y.numel(), unique_x.numel())
    order = 0
    while (1 << order) < max_dim:
        order += 1
    n_side = 1 << order

    x_numpy = x_dense.cpu().numpy()
    y_numpy = y_dense.cpu().numpy()
    h_dists_numpy = _compute_hilbert_distances(x_numpy, y_numpy, n_side)
    h_dists = torch.from_numpy(h_dists_numpy).to(device)

    sorted_indices = torch.argsort(h_dists, stable=True)
    total_patches = len(sorted_indices)

    if sample_num is None:
        final_indices = sorted_indices
        
    elif total_patches > sample_num:
        max_start = total_patches - sample_num
        start_idx = torch.randint(low=0, high=max_start + 1, size=(1,), device=device).item()
        end_idx = start_idx + sample_num
        
        final_indices = sorted_indices[start_idx : end_idx]
        
    else:
        final_indices = sorted_indices

    return coords_yx[final_indices], final_indices