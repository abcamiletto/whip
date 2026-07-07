import torch
from jaxtyping import Float
from torch import Tensor


@torch.no_grad()
def metrics(pred: Float[Tensor, "N T J 3"], gt: Float[Tensor, "N T J 3"]) -> dict[str, float]:
    pred = pred.reshape(-1, pred.shape[-2], 3)
    gt = gt.reshape(-1, gt.shape[-2], 3)
    root = root_align(pred, gt)
    norm = scale_align(pred, gt)
    error = torch.linalg.norm(root - gt, dim=-1)
    n_error = torch.linalg.norm(norm - gt, dim=-1)
    return {
        "mpjpe": error.mean().item() * 1000,
        "n-mpjpe": n_error.mean().item() * 1000,
    }


def root_align(pred: Float[Tensor, "N J 3"], gt: Float[Tensor, "N J 3"], root: int = 0) -> Float[Tensor, "N J 3"]:
    return pred + gt[:, root : root + 1] - pred[:, root : root + 1]


def scale_align(pred: Float[Tensor, "N J 3"], gt: Float[Tensor, "N J 3"], eps: float = 1e-8) -> Float[Tensor, "N J 3"]:
    pred_mean = pred.mean(1, keepdim=True)
    gt_mean = gt.mean(1, keepdim=True)
    pred_centered = pred - pred_mean
    gt_centered = gt - gt_mean
    scale = (pred_centered * gt_centered).sum((1, 2)) / (pred_centered.square().sum((1, 2)).clamp_min(eps))
    return scale[:, None, None] * pred_centered + gt_mean
