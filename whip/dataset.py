import functools
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float
from torch.utils.data import Dataset

from .skeleton import DATASET_KEYPOINTS, MODEL_KEYPOINTS

IMUS = ("watch_left", "watch_right", "phone_left", "phone_right")
TEST_SEQUENCES = {"test_actor00_seq1", "test_actor00_seq2"}
TEST_ACTIONS = {"test_archery", "test_play_volleyball", "test_tie_shoelaces"}


@dataclass(frozen=True)
class Item:
    path: Path
    sequence: str
    action: str
    frames: tuple[int, ...]


class WhipDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        sequence_length: int = 90,
        stride: int = 1,
        keypoints: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        if not (self.root / "actions.txt").exists():
            raise FileNotFoundError(f"expected the WHIP data/ folder, got {self.root}")
        self.sequence_length = sequence_length
        self.stride = stride
        self.keypoints = keypoints or MODEL_KEYPOINTS
        self.keypoint_ids = tuple(DATASET_KEYPOINTS.index(name) for name in self.keypoints)
        self.actions = (self.root / "actions.txt").read_text().splitlines()
        self.items = self.index(split)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.items[i]
        joints = load_joints(item.path, item.frames, self.keypoint_ids)
        imus = [load_imu(item.path, device, item.frames) for device in IMUS]
        vr_pose = load_pose(item.path, item.frames)
        transform = scene_transform(joints, self.keypoints)
        action = np.int64(self.actions.index(item.action))

        return {
            "joints": transform_joints(joints, transform),
            "imus": transform_imus(imus, transform),
            "vr_pose": transform_pose(vr_pose, transform),
            "insoles": load_insoles(item.path, item.frames),
            "action": action,
        }

    def index(self, split: str) -> list[Item]:
        items = []
        for sequence in sorted(path.name for path in self.root.iterdir() if (path / "actions").is_dir()):
            for action_path in sorted((self.root / sequence / "actions").glob("*.tar")):
                action = action_path.stem
                if not in_split(split, sequence, action):
                    continue
                n = int(read_tar_member(action_path, "n_frames.txt").decode())
                for start in range(0, n - self.sequence_length + 1, self.stride):
                    items.append(Item(action_path, sequence, action, tuple(range(start, start + self.sequence_length))))
        return items


def in_split(split: str, sequence: str, action: str) -> bool:
    test_sequence = sequence in TEST_SEQUENCES
    test_action = action in TEST_ACTIONS
    test = test_sequence or test_action
    train = not test
    unseen_actor = test_sequence and not test_action
    unseen_actions = test_action and not test_sequence
    unseen_both = test_sequence and test_action

    return {
        "train": train,
        "test": test,
        "unseen_actor": unseen_actor,
        "unseen_actions": unseen_actions,
        "unseen_both": unseen_both,
        "unseen_actor_or_actions": test,
    }[split]


def scene_transform(joints: Float[np.ndarray, "T J 3"], keypoints: list[str]) -> Float[np.ndarray, "4 4"]:
    root = keypoints.index("Hips")
    left = keypoints.index("LeftUpLeg")
    right = keypoints.index("RightUpLeg")
    center = joints[0, root].copy()
    center[1] = 0
    hip = joints[0, left] - joints[0, right]
    rotation = horizontal_rotation(hip)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ center
    return transform


def transform_joints(
    joints: Float[np.ndarray, "T J 3"], transform: Float[np.ndarray, "4 4"]
) -> Float[np.ndarray, "T J 3"]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (joints @ rotation.T + translation).astype(np.float32)


def transform_pose(pose: Float[np.ndarray, "T 4 4"], transform: Float[np.ndarray, "4 4"]) -> Float[np.ndarray, "T 4 4"]:
    return (transform[None] @ pose).astype(np.float32)


def transform_imus(
    imus: list[dict[str, Float[np.ndarray, "T ..."]]], transform: Float[np.ndarray, "4 4"]
) -> list[dict[str, Float[np.ndarray, "T ..."]]]:
    rotation = transform[:3, :3]
    for imu in imus:
        imu["orientation"] = rotation[None] @ imu["orientation"]
    return imus


def horizontal_rotation(hip: Float[np.ndarray, "3"]) -> Float[np.ndarray, "3 3"]:
    norm = np.linalg.norm(hip[[0, 2]])
    cos = hip[0] / norm if norm > 1e-8 else 1.0
    sin = hip[2] / norm if norm > 1e-8 else 0.0
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float32)


def load_joints(action_path: Path, frames: tuple[int, ...], keypoints: tuple[int, ...]) -> Float[np.ndarray, "T J 3"]:
    joints = load_npz_member(action_path, "body_tracking/joints_3D.npz")["translations"]
    return joints[list(frames)][:, keypoints]


def load_insoles(action_path: Path, frames: tuple[int, ...]) -> dict[str, Float[np.ndarray, "T 2 D"]]:
    data = load_npz_member(action_path, "insoles/readings.npz", allow_pickle=True)
    left, right = data["left"].item(), data["right"].item()
    frame_ids = list(frames)
    insoles = {}
    for key in left:
        left_values = left[key][frame_ids]
        right_values = right[key][frame_ids]
        insoles[key] = np.stack([left_values, right_values], axis=1)
    return insoles


def load_imu(action_path: Path, device: str, frames: tuple[int, ...]) -> dict[str, Float[np.ndarray, "T ..."]]:
    data = load_npz_member(action_path, f"imu/{device}.npz")
    frame_ids = list(frames)
    imu = {}
    for key in data:
        values = data[key][frame_ids]
        imu[key] = values.astype(np.float32)
    return imu


def load_pose(action_path: Path, frames: tuple[int, ...]) -> Float[np.ndarray, "T 4 4"]:
    data = load_npz_member(action_path, "vr/poses.npz")
    pose = np.repeat(np.eye(4, dtype=np.float32)[None], len(frames), axis=0)
    pose[:, :3, :3] = data["rotations"][list(frames)]
    pose[:, :3, 3] = data["translations"][list(frames)].squeeze()
    return pose


@functools.lru_cache(maxsize=32768)
def load_npz_member(action_path: Path, member: str, allow_pickle: bool = False) -> dict[str, Any]:
    with np.load(io.BytesIO(read_tar_member(action_path, member)), allow_pickle=allow_pickle) as data:
        return {key: data[key] for key in data.files}


def read_tar_member(action_path: Path, member: str) -> bytes:
    with tarfile.open(action_path) as tar:
        file = tar.extractfile(member)
        if file is None:
            raise FileNotFoundError(f"{member} not found in {action_path}")
        return file.read()
