import math
from functools import partial

import torch
import torch._dynamo
import torch.nn.functional as F
from jaxtyping import Float, Int
from timm.layers import Mlp
from torch import Tensor, nn

from .skeleton import MODEL_KEYPOINTS
from .types import Batch

torch._dynamo.config.cache_size_limit = 256
ROOT = MODEL_KEYPOINTS.index("Hips")


class Flow(nn.Module):
    def __init__(self, joints: int = 19, actions: int = 55, dim: int = 768, depth: int = 8, heads: int = 12) -> None:
        super().__init__()
        self.joints = joints
        self.label = LabelEmbedder(actions, 128, 0.5)
        self.insole = InsoleEmbedder(64)
        self.imu = ImuEmbedder(64)
        self.vr = VRPoseEmbedder(64)
        self.net = VelocityDiT(joints * 3, dim, 128, depth, heads)

    def loss(self, batch: Batch) -> Float[Tensor, ""]:
        x1 = encode_root_relative(batch["joints"])
        x0 = torch.randn_like(x1)
        t = beta_time(x1.shape[0], x1.device)
        xt = x0.lerp(x1, t[:, None, None, None])
        pred = self(
            xt,
            t,
            batch,
            sample_masks(x1.shape[0], x1.device) if self.training else masks_for("all", x1.shape[0], x1.device),
        )
        return (pred - (x1 - x0)).square().mean()

    def forward(
        self, x: Float[Tensor, "B T J 3"], t: Float[Tensor, " B "], batch: Batch, masks: dict
    ) -> Float[Tensor, "B T J 3"]:
        velocity = self.net(
            x.flatten(2),
            t,
            self.label(batch["action"] if masks["label"] else None),
            self.insole(**batch["insoles"]),
            [self.imu(**imu) for imu in batch["imus"]],
            self.vr(batch["vr_pose"]),
            masks,
        )
        return velocity.unflatten(2, (self.joints, 3))

    @torch.no_grad()
    def sample(self, batch: Batch, combo: str = "all") -> Float[Tensor, "B T J 3"]:
        x = torch.randn_like(batch["joints"])
        grid = ode_grid(x.device)
        masks = masks_for(combo, x.shape[0], x.device)
        for t0, t1 in zip(grid[:-1], grid[1:]):
            t = t0.expand(x.shape[0])
            x = x + (t1 - t0) * self(x, t, batch, masks)
        return decode_root_relative(x)


class VelocityDiT(nn.Module):
    def __init__(self, state: int, dim: int, cond: int, depth: int, heads: int) -> None:
        super().__init__()
        assert dim % heads == 0
        self.x = nn.Linear(state, dim - 192)
        self.t = Timestep(cond)
        self.rotary = Rotary(dim // heads)
        self.insole_proj = nn.Linear(64, 32)
        self.imu_proj = nn.ModuleList(nn.Linear(64, 32) for _ in range(4))
        self.vr_proj = nn.Linear(64, 32)
        self.insole_norm = nn.LayerNorm(64, bias=False)
        self.imu_norm = nn.ModuleList(nn.LayerNorm(64, bias=False) for _ in range(4))
        self.vr_norm = nn.LayerNorm(64, bias=False)
        self.null_insole = nn.Parameter(torch.zeros(1, 1, 32))
        self.null_imu = nn.Parameter(torch.zeros(1, 1, 32))
        self.null_vr = nn.Parameter(torch.zeros(1, 1, 32))
        self.blocks = nn.ModuleList(Block(dim, heads, cond, use_cross=i in {0, 3, 6, 7}) for i in range(depth))
        self.out = FinalLayer(dim, state, cond)

    def forward(
        self,
        x: Float[Tensor, "B T State"],
        t: Float[Tensor, " B "],
        label: Float[Tensor, "B Cond"],
        insole: Float[Tensor, "B T Sensor"],
        imus: list[Float[Tensor, "B T Sensor"]],
        vr: Float[Tensor, "B T Sensor"],
        masks: dict,
    ) -> Float[Tensor, "B T State"]:
        x = torch.cat([self.x(x), self.sensor_state(insole, imus, vr, masks)], -1)
        cond = self.t(t) + label
        rotary = self.rotary(x)
        for block in self.blocks:
            x = block(x, rotary, cond, insole, imus, vr, masks)
        return self.out(x, cond)

    def sensor_state(
        self,
        insole: Float[Tensor, "B T Sensor"],
        imus: list[Float[Tensor, "B T Sensor"]],
        vr: Float[Tensor, "B T Sensor"],
        masks: dict,
    ) -> Float[Tensor, "B T Slots"]:
        b, t = insole.shape[:2]
        slots = [self.masked(self.insole_proj(self.insole_norm(insole)), self.null_insole, masks["insole_mask"])]
        slots += [
            self.masked(proj(norm(imu)), self.null_imu, masks["imu_masks"][i])
            for i, (proj, norm, imu) in enumerate(zip(self.imu_proj, self.imu_norm, imus, strict=True))
        ]
        slots += [self.masked(self.vr_proj(self.vr_norm(vr)), self.null_vr, masks["vr_pose_mask"])]
        return torch.cat([slot.expand(b, t, -1) if slot.shape[:2] == (1, 1) else slot for slot in slots], -1)

    def masked(
        self, x: Float[Tensor, "B T Slot"], null: Float[Tensor, "1 1 Slot"], mask: Float[Tensor, " B "]
    ) -> Float[Tensor, "B T Slot"]:
        mask = mask[:, None, None]
        return x * mask + null.expand(x.shape[0], x.shape[1], -1) * (1 - mask)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, cond: int, use_cross: bool) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.use_cross = use_cross
        self.norm1 = nn.LayerNorm(dim, bias=False)
        self.qkv = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ada = zero(nn.Linear(cond, 6 * dim))
        self.norm2 = nn.LayerNorm(dim, bias=False)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=2 * dim,
            out_features=dim,
            act_layer=partial(nn.GELU, approximate="tanh"),
        )
        if use_cross:
            self.cross_ada = zero(nn.Linear(cond, 3 * dim))
            self.cross_norm = nn.LayerNorm(dim, bias=False)
            self.cross_insole = CrossAttention(dim, heads=3, ctx=64, hidden=192, window=7)
            self.cross_imu = nn.ModuleList(CrossAttention(dim, heads=3, ctx=64, hidden=192, window=7) for _ in range(4))
            self.cross_vr = CrossAttention(dim, heads=3, ctx=64, hidden=192, window=7)

    @torch.compile()
    def forward(
        self,
        x: Float[Tensor, "B T Model"],
        rotary: tuple,
        cond: Float[Tensor, "B Cond"],
        insole: Float[Tensor, "B T Sensor"],
        imus: list[Float[Tensor, "B T Sensor"]],
        vr: Float[Tensor, "B T Sensor"],
        masks: dict,
    ) -> Float[Tensor, "B T Model"]:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(cond).unsqueeze(1).chunk(6, -1)
        y = self.norm1(x) * (1 + scale_a) + shift_a
        q, k, v = [layer(y).view(y.shape[0], y.shape[1], self.heads, self.head_dim) for layer in self.qkv]
        cos, sin = rotary
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        y = y.transpose(1, 2).flatten(2)
        x = x + gate_a * F.dropout(self.attn_out(y), p=0.1, training=self.training)
        if self.use_cross:
            shift_c, scale_c, gate_c = self.cross_ada(cond).unsqueeze(1).chunk(3, -1)
            y = self.cross_norm(x) * (1 + scale_c) + shift_c
            x = x + gate_c * F.dropout(self.cross(y, insole, imus, vr, masks), p=0.1, training=self.training)
        y = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * F.dropout(self.mlp(y), p=0.1, training=self.training)

    def cross(
        self,
        x: Float[Tensor, "B T Model"],
        insole: Float[Tensor, "B T Sensor"],
        imus: list[Float[Tensor, "B T Sensor"]],
        vr: Float[Tensor, "B T Sensor"],
        masks: dict,
    ) -> Float[Tensor, "B T Model"]:
        parts = [self.cross_insole(x, insole) * masks["insole_mask"][:, None, None]]
        parts += [
            attn(x, imu) * masks["imu_masks"][i][:, None, None]
            for i, (attn, imu) in enumerate(zip(self.cross_imu, imus, strict=True))
        ]
        parts += [self.cross_vr(x, vr) * masks["vr_pose_mask"][:, None, None]]
        present = masks["insole_mask"] + masks["vr_pose_mask"] + sum(masks["imu_masks"])
        return torch.stack(parts).sum(0) / present.clamp(min=1).sqrt()[:, None, None]


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, ctx: int, hidden: int, window: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(dim, hidden, bias=False)
        self.k = nn.Linear(ctx, hidden, bias=False)
        self.v = nn.Linear(ctx, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)
        self.out.weight.data.div_(2.0)
        self.rotary = Rotary(self.head_dim)
        self.mod = SensorModulation(ctx, dim)
        self.window = window
        self.mask = None

    def forward(self, x: Float[Tensor, "B T Model"], ctx: Float[Tensor, "B T Sensor"]) -> Float[Tensor, "B T Model"]:
        gamma, beta, alpha = self.mod(ctx)
        x = x * (1 + gamma) + beta
        q = self.q(x).view(x.shape[0], x.shape[1], self.heads, self.head_dim)
        k = self.k(ctx).view(x.shape[0], x.shape[1], self.heads, self.head_dim)
        v = self.v(ctx).view(x.shape[0], x.shape[1], self.heads, self.head_dim)
        cos, sin = self.rotary(k)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.mask is None or self.mask.shape[0] != x.shape[1] or self.mask.device != x.device:
            i = torch.arange(x.shape[1], device=x.device)
            self.mask = (i[:, None] - i[None]).abs() > self.window
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=self.mask)
        return alpha * self.out(y.transpose(1, 2).flatten(2))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out: int, cond: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias=False)
        self.linear = zero(nn.Linear(dim, out))
        self.ada = zero(nn.Linear(cond, 2 * dim))

    def forward(self, x: Float[Tensor, "B T Model"], cond: Float[Tensor, "B Cond"]) -> Float[Tensor, "B T State"]:
        shift, scale = self.ada(cond).unsqueeze(1).chunk(2, -1)
        return self.linear(self.norm(x) * (1 + scale) + shift)


class InsoleEmbedder(nn.Module):
    dims = {"pressure": 16, "acceleration": 3, "angular_vel": 3, "force": 1, "cop": 2}
    hidden = {"pressure": 64, "acceleration": 16, "angular_vel": 16, "force": 16, "cop": 16}

    def __init__(self, out: int) -> None:
        super().__init__()
        self.mlps = nn.ModuleDict()
        for key, dim in self.dims.items():
            features = self.hidden[key]
            self.mlps[key] = Mlp(
                in_features=dim,
                hidden_features=features,
                out_features=features,
                norm_layer=nn.LayerNorm,
                drop=0.1,
            )
        self.out = nn.Linear(sum(self.hidden.values()), out // 2)
        self.side = nn.Parameter(torch.randn(2, out // 2))

    def forward(self, **data: Float[Tensor, "B T ..."]) -> Float[Tensor, "B T Sensor"]:
        b, t = data["pressure"].shape[:2]
        sides = []
        for side in range(2):
            x = torch.cat([self.mlps[k](data[k][:, :, side].reshape(-1, dim)) for k, dim in self.dims.items()], -1)
            sides.append((self.out(x) + self.side[side]).reshape(b, t, -1))
        return torch.cat(sides, -1)


class ImuEmbedder(nn.Module):
    dims = {"orientation": 9, "angular_velocity": 3, "acceleration": 3}
    hidden = {"orientation": 64, "angular_velocity": 16, "acceleration": 16}

    def __init__(self, out: int) -> None:
        super().__init__()
        self.mlps = nn.ModuleDict()
        for key, dim in self.dims.items():
            features = self.hidden[key]
            self.mlps[key] = Mlp(
                in_features=dim,
                hidden_features=features,
                out_features=features,
                norm_layer=nn.LayerNorm,
                drop=0.1,
            )
        self.out = nn.Linear(sum(self.hidden.values()), out)

    def forward(self, **data: Float[Tensor, "B T ..."]) -> Float[Tensor, "B T Sensor"]:
        b, t = data["orientation"].shape[:2]
        data = data | {"orientation": data["orientation"].reshape(b, t, 9)}
        x = torch.cat([self.mlps[k](data[k].reshape(-1, dim)) for k, dim in self.dims.items()], -1)
        return self.out(x).reshape(b, t, -1)


class VRPoseEmbedder(nn.Module):
    def __init__(self, out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(9)
        self.out = nn.Linear(12, out)

    def forward(self, pose: Float[Tensor, "B T 4 4"]) -> Float[Tensor, "B T Sensor"]:
        b, t = pose.shape[:2]
        rot = self.norm(pose[:, :, :3, :3].reshape(-1, 9))
        trans = pose[:, :, :3, 3].reshape(-1, 3)
        return self.out(torch.cat([rot, trans], -1)).reshape(b, t, -1)


class LabelEmbedder(nn.Module):
    def __init__(self, classes: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(classes + 1, dim)
        self.classes = classes
        self.dropout = dropout
        self.register_buffer("seen", torch.zeros(classes, dtype=torch.bool))

    def forward(self, labels: Int[Tensor, " B "] | None) -> Float[Tensor, "B Cond"]:
        if labels is None:
            labels = torch.full((1,), self.classes, device=self.embed.weight.device, dtype=torch.long)
            return self.embed(labels)
        labels = labels.clone()
        if self.training:
            self.seen[labels] = True
            labels[torch.rand(labels.shape, device=labels.device) < self.dropout] = self.classes
        else:
            labels[~self.seen[labels]] = self.classes
        return self.embed(labels)


class SensorModulation(nn.Module):
    def __init__(self, ctx: int, out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx, max(32, out // 4)),
            nn.GELU(approximate="tanh"),
            zero(nn.Linear(max(32, out // 4), 2 * out + 1)),
        )
        self.out = out

    def forward(
        self, ctx: Float[Tensor, "B T Sensor"]
    ) -> tuple[Float[Tensor, "B T Model"], Float[Tensor, "B T Model"], Float[Tensor, "B T 1"]]:
        gamma, beta, alpha = self.net(ctx).split([self.out, self.out, 1], -1)
        return gamma, beta, alpha.sigmoid()


class Timestep(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Float[Tensor, " B "]) -> Float[Tensor, "B Cond"]:
        freqs = torch.exp(-math.log(10000) * torch.arange(128, device=t.device, dtype=torch.float32) / 128)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], -1))


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 512) -> None:
        super().__init__()
        self.register_buffer("inv_freq", 1 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.cache = None

    def forward(self, x: Float[Tensor, "B T ..."]) -> tuple:
        seq = x.shape[1]
        if self.cache is None or self.cache[0].shape[1] != seq or self.cache[0].device != x.device:
            t = torch.arange(seq, device=x.device).type_as(self.inv_freq)
            emb = torch.cat([torch.einsum("i,j->ij", t, self.inv_freq)] * 2, -1)
            self.cache = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
        return self.cache


def encode_root_relative(joints: Float[Tensor, "B T J 3"]) -> Float[Tensor, "B T J 3"]:
    state = joints.clone()
    root = joints[:, :, ROOT].clone()
    state[..., [0, 2]] -= root[:, :, None, [0, 2]]
    state[:, :, ROOT, [0, 2]] = root[:, :, [0, 2]]
    return state


def decode_root_relative(state: Float[Tensor, "B T J 3"]) -> Float[Tensor, "B T J 3"]:
    joints = state.clone()
    root_xz = state[:, :, ROOT, [0, 2]].clone()
    joints[..., [0, 2]] += root_xz[:, :, None]
    joints[:, :, ROOT, [0, 2]] = root_xz
    return joints


def zero(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def apply_rotary(
    x: Float[Tensor, "B T H Head"], cos: Float[Tensor, "1 T 1 Head"], sin: Float[Tensor, "1 T 1 Head"]
) -> Float[Tensor, "B T H Head"]:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos[..., : x.shape[-1]] + rotated * sin[..., : x.shape[-1]]


def beta_time(batch_size: int, device: torch.device) -> Float[Tensor, " B "]:
    return torch.distributions.Beta(1.84, 2.16).sample((batch_size,)).to(device)


def ode_grid(device: torch.device) -> Float[Tensor, "7"]:
    return torch.tensor(
        [0.0, 0.2995325352459066, 0.4330749999462677, 0.5472638795355319, 0.658076491156192, 0.7790903552254396, 1.0],
        device=device,
    )


def sample_masks(batch_size: int, device: torch.device) -> dict:
    masks = torch.rand(batch_size, 6, device=device) < 0.5
    return {
        "label": True,
        "insole_mask": masks[:, 0].float(),
        "vr_pose_mask": masks[:, 1].float(),
        "imu_masks": [masks[:, i].float() for i in range(2, 6)],
    }


def masks_for(combo: str, batch_size: int, device: torch.device) -> dict:
    masks = torch.zeros(batch_size, 6, device=device)
    sensor_ids = {
        "insoles": 0,
        "vr_pose": 1,
        "watch_left": 2,
        "watch_right": 3,
        "phone_left": 4,
        "phone_right": 5,
    }

    sensors = {sensor.strip() for sensor in combo.split(",")}
    label = "all" in sensors or "label" in sensors
    if "all" in sensors:
        masks[:] = 1
    else:
        for sensor in sensors - {"label"}:
            masks[:, sensor_ids[sensor]] = 1
    return {
        "label": label,
        "insole_mask": masks[:, 0],
        "vr_pose_mask": masks[:, 1],
        "imu_masks": [masks[:, i] for i in range(2, 6)],
    }
