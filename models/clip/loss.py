from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, case_ids=None, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale) 

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        # ------------------------------------------------------------
        # Top-1 statistics (Image→Text and Text→Image)
        # ------------------------------------------------------------
        with torch.no_grad():
            batch_img2txt_top1 = (logits_per_image.argmax(dim=1) == labels).float().mean()
            batch_txt2img_top1 = (logits_per_text.argmax(dim=1) == labels).float().mean()

        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "img2txt_top1": batch_img2txt_top1,
                "txt2img_top1": batch_txt2img_top1,
            }
        return total_loss


class MultiPosClipLoss(ClipLoss):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )

    def forward(self, image_features, text_features, logit_scale, case_ids, output_dict=False):
        device = image_features.device

        # Start with local features and case_ids
        all_image_features = image_features
        all_text_features = text_features
        all_case_ids = case_ids

        # Gather features and case_ids from all GPUs if distributed
        if self.world_size > 1:
            # All-gather for tensors
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )
            # All-gather for python objects (case_ids)
            # This requires a slightly different utility
            gathered_case_ids = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_case_ids, case_ids)
            all_case_ids = [item for sublist in gathered_case_ids for item in sublist]

        # Normalize features
        all_image_features = F.normalize(all_image_features, dim=-1)
        all_text_features = F.normalize(all_text_features, dim=-1)
        
        # Calculate similarity matrix
        sim = all_image_features @ all_text_features.T * logit_scale
        
        # NOTE: The use of hash() is not guaranteed to be consistent across Python processes
        # or different runs, due to hash randomization. This may affect reproducibility.
        # However, within a single distributed training run, the positive pair mask
        # generated should be consistent across all ranks because each rank hashes
        # the same gathered list of case_ids.
        # TODO: For robust reproducibility, consider replacing hash() with a deterministic
        # mapping from case_id strings to unique integer IDs.
        case_ids_tensor = torch.tensor(
            [hash(cid) for cid in all_case_ids], 
            device=device,
            dtype=torch.long
        )
        pos_mask = (case_ids_tensor[:, None] == case_ids_tensor[None, :]).float()
        # pos_mask.fill_diagonal_(0) # Exclude self-similarity - REMOVED FOR CORRECTNESS

        exp_sim = torch.exp(sim)
        log_prob = -torch.log((exp_sim * pos_mask).sum(1) / exp_sim.sum(1) )
        
        # Check for NaNs which can happen if a sample has no positive pairs
        valid_samples = pos_mask.sum(1) > 0
        if valid_samples.sum() == 0:
            loss_i2t = (sim * 0.).sum()
        else:
            loss_i2t = log_prob[valid_samples].mean()

        # Symmetrical loss for text-to-image
        sim_t2i = sim.T
        exp_sim_t2i = torch.exp(sim_t2i)
        log_prob_t2i = -torch.log((exp_sim_t2i * pos_mask).sum(1) / exp_sim_t2i.sum(1))
        
        if valid_samples.sum() == 0:
            loss_t2i = (sim_t2i * 0.).sum()
        else:
            loss_t2i = log_prob_t2i[valid_samples].mean()

        total_loss = (loss_i2t + loss_t2i) / 2

        # Calculate top-1 accuracy for logging
        with torch.no_grad():
            sim_diag = torch.diag(sim)
            # Create a mask for valid negatives (excluding self and other positives)
            neg_mask = 1 - pos_mask
            
            # For each image, check if the highest similarity (excluding other positives) is the diagonal (self)
            masked_sim_i2t = sim * neg_mask + (1 - neg_mask) * -1e9
            best_neg_sim_i2t = masked_sim_i2t.max(dim=1).values
            img2txt_top1 = (sim_diag > best_neg_sim_i2t).float().mean()
            
            # Symmetrical for text-to-image
            masked_sim_t2i = sim_t2i * neg_mask + (1 - neg_mask) * -1e9
            best_neg_sim_t2i = masked_sim_t2i.max(dim=1).values
            txt2img_top1 = (sim_diag > best_neg_sim_t2i).float().mean()

        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "img2txt_top1": img2txt_top1,
                "txt2img_top1": txt2img_top1,
            }

        return total_loss



class WSIClipLoss(nn.Module):
    """
    Contrastive loss for MI-Zero style WSI-CLIP pre-training.

    Each whole-slide image (WSI) consists of multiple patches but shares **one**
    caption.  The loss combines two InfoNCE terms:

    1. Patch-to-Text (P→T): every patch must identify its corresponding text
       among all texts in the global batch.
    2. Text-to-WSI (T→W): every caption must identify the mean-pooled visual
       embedding of its own WSI among all slide-level embeddings.

    Patch embeddings can vary in number per slide and therefore are kept **local**
    to each GPU.  Fixed-size tensors (text features & WSI features) are gathered
    across ranks so that negatives come from the whole world, exactly mirroring
    standard `ClipLoss` behaviour.
    """

    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ) -> None:
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # Cache for ground-truth label tensors (to avoid reallocations)
        self.prev_num_logits = 0
        self.labels: dict[torch.device, torch.Tensor] = {}

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _get_ground_truth(self, device: torch.device, num: int) -> torch.Tensor:
        """Return tensor `[0, 1, …, num-1]` (optionally offset by rank)."""
        if self.prev_num_logits != num or device not in self.labels:
            labels = torch.arange(num, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num
        else:
            labels = self.labels[device]
        return labels

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        patch_features_list: list[torch.Tensor],  # len = B, each [n_i, D]
        text_features: torch.Tensor,              # [B, D]
        logit_scale: torch.Tensor,                # scalar (already exp() outside)
        output_dict: bool = False,
    ):
        device = text_features.device
        B = text_features.size(0)

        # 1. Slide-level (WSI) embeddings via mean pooling across patches
        wsi_features = torch.stack(
            [patch.mean(dim=0) for patch in patch_features_list],
            dim=0,
        )  # [B, D]

        # 2. Flatten all patch embeddings and record their parent slide id
        patch_features = torch.cat(patch_features_list, dim=0)  # [P, D]
        patch_to_slide = torch.cat(
            [
                torch.full((patch.shape[0],), idx, device=device, dtype=torch.long)
                for idx, patch in enumerate(patch_features_list)
            ],
            dim=0,
        )  # [P]

        # 3. Gather fixed-size tensors (texts & WSI embeddings) across ranks
        all_text, all_wsi = text_features, wsi_features
        if self.world_size > 1:
            all_text, all_wsi = gather_features(
                text_features,
                wsi_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

        # 4. Compute similarity logits
        #    Patch → Text : [P, N_text]
        logits_p2t = logit_scale * patch_features @ all_text.T
        #    Text  → WSI  : [B, N_wsi]
        logits_t2w = logit_scale * text_features @ all_wsi.T

        # 5. Prepare ground-truth indices
        if self.world_size > 1:
            patch_labels = patch_to_slide + B * self.rank
        else:
            patch_labels = patch_to_slide

        wsi_labels = self._get_ground_truth(device, B)

        # 6. Compute cross-entropy losses
        loss_p2t = F.cross_entropy(logits_p2t, patch_labels)
        loss_t2w = F.cross_entropy(logits_t2w, wsi_labels)

        total_loss = 0.5 * (loss_p2t + loss_t2w)

        # ------------------------------------------------------------
        # 7. Calculate Batch-Patch Top-1 and Text-WSI Top-1 accuracy (only for statistics, not for gradient)
        # ------------------------------------------------------------
        with torch.no_grad():
            batch_patch_top1 = (logits_p2t.argmax(dim=1) == patch_labels).float().mean()
            batch_text_top1 = (logits_t2w.argmax(dim=1) == wsi_labels).float().mean()

        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "patch2text_loss": loss_p2t,
                "text2wsi_loss": loss_t2w,
                # Accuracy metrics
                "batch_patch_top1": batch_patch_top1,
                "batch_text_top1": batch_text_top1,
            }
        return total_loss



class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss
