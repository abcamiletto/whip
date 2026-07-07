from typing import TypedDict

import torch
from jaxtyping import Float, Int
from torch import Tensor


class InsoleBatch(TypedDict):
    pressure: Float[Tensor, "B T 2 16"]
    acceleration: Float[Tensor, "B T 2 3"]
    angular_vel: Float[Tensor, "B T 2 3"]
    force: Float[Tensor, "B T 2 1"]
    cop: Float[Tensor, "B T 2 2"]


class ImuBatch(TypedDict):
    orientation: Float[Tensor, "B T 3 3"]
    angular_velocity: Float[Tensor, "B T 3"]
    acceleration: Float[Tensor, "B T 3"]


class Batch(TypedDict):
    joints: Float[Tensor, "B T J 3"]
    insoles: InsoleBatch
    imus: list[ImuBatch]
    vr_pose: Float[Tensor, "B T 4 4"]
    action: Int[Tensor, " B "]


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return {
        "joints": batch["joints"].to(device, non_blocking=True),
        "insoles": {
            "pressure": batch["insoles"]["pressure"].to(device, non_blocking=True),
            "acceleration": batch["insoles"]["acceleration"].to(device, non_blocking=True),
            "angular_vel": batch["insoles"]["angular_vel"].to(device, non_blocking=True),
            "force": batch["insoles"]["force"].to(device, non_blocking=True),
            "cop": batch["insoles"]["cop"].to(device, non_blocking=True),
        },
        "imus": [
            {
                "orientation": imu["orientation"].to(device, non_blocking=True),
                "angular_velocity": imu["angular_velocity"].to(device, non_blocking=True),
                "acceleration": imu["acceleration"].to(device, non_blocking=True),
            }
            for imu in batch["imus"]
        ],
        "vr_pose": batch["vr_pose"].to(device, non_blocking=True),
        "action": batch["action"].to(device, non_blocking=True),
    }
