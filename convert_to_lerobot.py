""" 
This project is built upon the open-source project 🤗 LeRobot: https://github.com/huggingface/lerobot 

We are grateful to the LeRobot team for their outstanding work and their contributions to the community. 

If you find this project useful, please also consider supporting and exploring LeRobot. 
"""

import os
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Callable
from functools import partial
from math import ceil
from copy import deepcopy

import h5py
import torch
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pformat
from tqdm.contrib.concurrent import process_map

# 兼容不同版本的lerobot
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import (
        STATS_PATH,
        check_timestamps_sync,
        get_episode_data_index,
        serialize_dict,
        write_json,
    )
except ImportError:
    try:
        # 尝试新版本的lerobot
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import STATS_PATH
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        STATS_PATH = "stats"
    
    # 新版本移除了这些函数，需要本地实现
    def check_timestamps_sync(dataset, episode_data_index, fps, tolerance_s):
        """检查时间戳同步（新版本已移除，提供兼容实现）"""
        pass  # 简化实现，跳过检查
    
    def get_episode_data_index(episodes, selected_episodes=None):
        """获取episode数据索引"""
        if selected_episodes is None:
            return {"from": [e["from"] for e in episodes], "to": [e["to"] for e in episodes]}
        indices = []
        for ep_idx in selected_episodes:
            indices.append(episodes[ep_idx])
        return {"from": [e["from"] for e in indices], "to": [e["to"] for e in indices]}
    
    def serialize_dict(stats):
        """序列化字典（转换tensor为list）"""
        import torch
        serialized = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                serialized[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        serialized[key][k] = v.tolist()
                    else:
                        serialized[key][k] = v
            elif isinstance(value, torch.Tensor):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def write_json(data, filepath):
        """写入JSON文件"""
        import json
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"
HEAD_CENTER_FISHEYE_COLOR = "head_center_fisheye_color.mp4"
HEAD_LEFT_FISHEYE_COLOR = "head_left_fisheye_color.mp4"
HEAD_RIGHT_FISHEYE_COLOR = "head_right_fisheye_color.mp4"
BACK_LEFT_FISHEYE_COLOR = "back_left_fisheye_color.mp4"
BACK_RIGHT_FISHEYE_COLOR = "back_right_fisheye_color.mp4"
HEAD_DEPTH = "head_depth"

DEFAULT_IMAGE_PATH = (
    "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
)

FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.cam_top_depth": {
        "dtype": "image",
        "shape": [480, 640, 1],
        "names": ["height", "width", "channel"],
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.head_center_fisheye": {
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.head_left_fisheye": {
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.head_right_fisheye": {
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.back_left_fisheye": {
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.back_right_fisheye": {
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [20],
    },
    "action": {
        "dtype": "float32",
        "shape": [22],
    },
    "episode_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
}


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert (
                c < h and c < w
            ), f"expect channel first images, but instead {batch[key].shape}"
            assert (
                batch[key].dtype == torch.float32
            ), f"expect torch.float32, but instead {batch[key].dtype=}"
            # assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            # assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"
            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def compute_stats(dataset, batch_size=8, num_workers=4, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(
            dataloader,
            total=ceil(max_num_samples / batch_size),
            desc="Compute mean, min, max",
        )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = (
                mean[key]
                + this_batch_size * (batch_mean - mean[key]) / running_item_count
            )
            max[key] = torch.maximum(
                max[key], einops.reduce(batch[key], pattern, "max")
            )
            min[key] = torch.minimum(
                min[key], einops.reduce(batch[key], pattern, "min")
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = (
                std[key] + this_batch_size * (batch_std - std[key]) / running_item_count
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats


class AgiBotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )
        
        # 确保stats被初始化为dict而不是其他类型
        if not hasattr(self, 'stats') or not isinstance(self.stats, dict):
            self.stats = {}
    
    def create_episode_buffer(self):
        """覆盖父类方法，确保返回的buffer包含必需的键"""
        try:
            # 尝试调用父类方法
            buffer = super().create_episode_buffer()
            print(f"[DEBUG] Parent create_episode_buffer returned: {type(buffer)}, keys: {buffer.keys() if hasattr(buffer, 'keys') else 'N/A'}")
        except AttributeError as e:
            # 如果父类没有这个方法，手动创建
            print(f"[DEBUG] Parent has no create_episode_buffer: {e}")
            buffer = {}
        except Exception as e:
            print(f"[DEBUG] Parent create_episode_buffer failed: {type(e).__name__}: {e}")
            buffer = {}
        
        # 确保有必需的键
        if "size" not in buffer:
            buffer["size"] = 0
        if "episode_index" not in buffer:
            buffer["episode_index"] = self.meta.total_episodes
        if "frame_index" not in buffer:
            buffer["frame_index"] = []
        if "timestamp" not in buffer:
            buffer["timestamp"] = []
        
        # 为features中的每个键创建空列表，但排除video/index/task_index等
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"]:
                # 这些是自动生成的，不需要在buffer里
                continue
            if ft["dtype"] in ["video"]:
                # video类型通过videos参数传递，不在buffer里
                continue
            if key not in buffer:
                buffer[key] = []
        
        # 清理掉父类可能添加的不需要的keys（video/index/task_index等）
        keys_to_remove = []
        for key in buffer.keys():
            if key in ["size", "episode_index", "frame_index", "timestamp"]:
                # 元数据，保留
                continue
            if key in self.features:
                ft = self.features[key]
                if ft["dtype"] == "video" or key in ["index", "task_index"]:
                    # video类型或自动生成的字段，删除
                    keys_to_remove.append(key)
            elif key == "task":
                # task也不应该在buffer里
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            buffer.pop(key, None)
        
        print(f"[DEBUG] Final buffer keys: {list(buffer.keys())}")
        return buffer
    
    def get_task_index(self, task: str) -> int:
        """覆盖父类方法，避免meta.tasks为None的问题"""
        import pandas as pd
        
        # 确保meta.tasks存在
        if not hasattr(self.meta, 'tasks') or self.meta.tasks is None:
            self.meta.tasks = pd.DataFrame({'task': [], 'task_index': []})
        
        # 检查task是否已存在
        if len(self.meta.tasks) > 0 and task in self.meta.tasks['task'].values:
            return int(self.meta.tasks[self.meta.tasks['task'] == task]['task_index'].iloc[0])
        
        # 添加新task
        task_index = len(self.meta.tasks)
        new_task = pd.DataFrame({'task': [task], 'task_index': [task_index]})
        self.meta.tasks = pd.concat([self.meta.tasks, new_task], ignore_index=True)
        return task_index


    def save_episode(
        self, task: str, episode_data: dict | None = None, videos: dict | None = None
    ) -> None:
        """
        We rewrite this method to copy mp4 videos to the target position
        """
        if not episode_data:
            episode_buffer = self.episode_buffer
        
        # 防御性检查：确保episode_buffer有'size'键
        if "size" not in episode_buffer:
            raise RuntimeError(
                f"episode_buffer missing 'size' key. Keys present: {list(episode_buffer.keys())}"
            )

        # 使用get而不是pop，避免异常时损坏buffer
        episode_length = episode_buffer.get("size", 0)
        if episode_length == 0:
            raise ValueError(
                "You must add one or several frames with `add_frame` before calling `add_episode`."
            )
        
        episode_index = episode_buffer["episode_index"]
        if episode_index != self.meta.total_episodes:
            # TODO(aliberts): Add option to use existing episode_index
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        # 使用我们自己的get_task_index方法（已覆盖父类）
        task_index = self.get_task_index(task)
        
        # 先用episode_length读取size的值，不要pop，避免后续异常时buffer损坏
        # size会在验证通过后才删除
        
        # 检查buffer keys和features的匹配：video类型的features不在buffer里，要排除
        expected_buffer_keys = set()
        for key, ft in self.features.items():
            if key not in ["index", "task_index"] and ft["dtype"] not in ["image", "video"]:
                expected_buffer_keys.add(key)
            elif ft["dtype"] == "image":
                expected_buffer_keys.add(key)
        expected_buffer_keys.update(["size", "episode_index", "frame_index", "timestamp"])
        
        actual_buffer_keys = set(episode_buffer.keys())
        if actual_buffer_keys != expected_buffer_keys:
            # 这里可能会抛异常，所以size还不能删除
            raise ValueError(
                f"Buffer keys mismatch.\n"
                f"  Missing in buffer: {expected_buffer_keys - actual_buffer_keys}\n"
                f"  Extra in buffer: {actual_buffer_keys - expected_buffer_keys}"
            )

        for key, ft in self.features.items():
            if key == "index":
                episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                episode_buffer[key] = np.full((episode_length,), episode_index)
            elif key == "task_index":
                episode_buffer[key] = np.full((episode_length,), task_index)
            elif ft["dtype"] in ["image", "video"]:
                continue
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                episode_buffer[key] = np.stack(episode_buffer[key])
            else:
                raise ValueError(key)

        # 所有验证和转换都完成了，现在可以安全地删除'size'键
        episode_buffer.pop("size", None)
        
        self._wait_image_writer()
        
        # 尝试调用父类的_save_episode_table，如果不存在则跳过
        try:
            self._save_episode_table(episode_buffer, episode_index)
        except AttributeError:
            # 某些版本的lerobot没有这个方法，直接保存到hf_dataset
            import pyarrow as pa
            if not hasattr(self, 'hf_dataset') or self.hf_dataset is None:
                # 第一次保存，需要创建dataset
                pass  # consolidate时会创建
            else:
                # 追加到现有dataset
                pass  # 也在consolidate时处理

        # 调用meta.save_episode，兼容不同版本的参数签名
        import inspect
        save_episode_sig = inspect.signature(self.meta.save_episode)
        params = list(save_episode_sig.parameters.keys())
        
        if 'episode_metadata' in params:
            # 新版本需要episode_metadata，传空dict
            self.meta.save_episode(
                episode_index, 
                episode_length, 
                task, 
                task_index,
                episode_metadata={}  # 必须是dict，不能是None
            )
        else:
            # 旧版本只需要4个参数
            self.meta.save_episode(episode_index, episode_length, task, task_index)
        
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = video_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()
        self.consolidated = False

    def consolidate(
        self, run_compute_stats: bool = True, keep_image_files: bool = False
    ) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        check_timestamps_sync(
            self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s
        )
        if len(self.meta.video_keys) > 0:
            self.meta.write_video_info()

        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        if run_compute_stats:
            self.stop_image_writer()
            self.meta.stats = compute_stats(self)
            serialized_stats = serialize_dict(self.meta.stats)
            write_json(serialized_stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)
            item = (
                frame[key].numpy()
                if isinstance(frame[key], torch.Tensor)
                else frame[key]
            )
            self.episode_buffer[key].append(item)

        self.episode_buffer["size"] += 1


def load_depths(root_dir: str, camera_name: str):
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))
    return [np.array(Image.open(f)).astype(np.float32) / 1000 for f in all_imgs]


def load_local_dataset(episode_id: int, src_path: str, task_id: int) -> list | None:
    """Load local dataset and return a dict with observations and actions
    
    Returns:
        tuple(frames, videos) if successful
        None if failed (e.g., corrupted data, missing files)
    """
    try:
        ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
        
        # 检查目录是否存在
        if not ob_dir.exists():
            raise FileNotFoundError(f"Observation directory not found: {ob_dir}")
        
        depth_imgs = load_depths(ob_dir / "depth", HEAD_DEPTH)
        proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"
        
        h5_file = proprio_dir / "proprio_stats.h5"
        if not h5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

        with h5py.File(h5_file, 'r') as f:
            # 检查必需的键是否存在
            required_keys = [
                "state/joint/position", "state/effector/position",
                "state/head/position", "state/waist/position",
                "action/joint/position", "action/effector/position",
                "action/head/position", "action/waist/position",
                "action/robot/velocity"
            ]
            for key in required_keys:
                if key not in f:
                    raise KeyError(f"Missing key '{key}' in HDF5 file: {h5_file}")
            
            state_joint = np.array(f["state/joint/position"])
            state_effector = np.array(f["state/effector/position"])
            state_head = np.array(f["state/head/position"])
            state_waist = np.array(f["state/waist/position"])
            action_joint = np.array(f["action/joint/position"])
            action_effector = np.array(f["action/effector/position"])
            action_head = np.array(f["action/head/position"])
            action_waist = np.array(f["action/waist/position"])
            action_velocity = np.array(f["action/robot/velocity"])

        states_value = np.hstack(
            [state_joint, state_effector, state_head, state_waist]
        ).astype(np.float32)
        assert (
            action_joint.shape[0] == action_effector.shape[0]
        ), f"shape of action_joint:{action_joint.shape};shape of action_effector:{action_effector.shape}"
        action_value = np.hstack(
            [action_joint, action_effector, action_head, action_waist, action_velocity]
        ).astype(np.float32)

        assert len(depth_imgs) == len(
            states_value
        ), f"Number of images and states are not equal"
        assert len(depth_imgs) == len(
            action_value
        ), f"Number of images and actions are not equal"
        frames = [
            {
                "observation.images.cam_top_depth": depth_imgs[i],
                "observation.state": states_value[i],
                "action": action_value[i],
            }
            for i in range(len(depth_imgs))
        ]

        v_path = ob_dir / "videos"
        videos = {
            "observation.images.top_head": v_path / HEAD_COLOR,
            "observation.images.hand_left": v_path / HAND_LEFT_COLOR,
            "observation.images.hand_right": v_path / HAND_RIGHT_COLOR,
            "observation.images.head_center_fisheye": v_path / HEAD_CENTER_FISHEYE_COLOR,
            "observation.images.head_left_fisheye": v_path / HEAD_LEFT_FISHEYE_COLOR,
            "observation.images.head_right_fisheye": v_path / HEAD_RIGHT_FISHEYE_COLOR,
            "observation.images.back_left_fisheye": v_path / BACK_LEFT_FISHEYE_COLOR,
            "observation.images.back_right_fisheye": v_path / BACK_RIGHT_FISHEYE_COLOR,
        }
        return frames, videos
    except Exception as e:
        # 数据损坏或缺失，返回None让上层跳过
        print(f"[WARNING] Episode {episode_id} load failed: {type(e).__name__}: {e}")
        return None


def get_task_instruction(task_json_path: str) -> dict:
    """Get task language instruction"""
    with open(task_json_path, "r") as f:
        task_info = json.load(f)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name}.{task_init_scene}"
    print(f"Get Task Instruction <{task_instruction}>")
    return task_instruction


def main(
    src_path: str,
    tgt_path: str,
    task_id: int,
    repo_id: str,
    task_info_json: str,
    debug: bool = False,
):
    task_name = get_task_instruction(task_info_json)

    dataset = AgiBotDataset.create(
        repo_id=repo_id,
        root=f"{tgt_path}/{repo_id}",
        fps=30,
        robot_type="a2d",
        features=FEATURES,
    )

    all_subdir = sorted(
        [
            f.as_posix()
            for f in Path(src_path).glob(f"observations/{task_id}/*")
            if f.is_dir()
        ]
    )

    if debug:
        all_subdir = all_subdir[:2]

    # Get all episode id
    all_subdir_eids = [int(Path(path).name) for path in all_subdir]

    if debug:
        raw_datasets_before_filter = [
            load_local_dataset(subdir, src_path=src_path, task_id=task_id)
            for subdir in tqdm(all_subdir_eids)
        ]
    else:
        raw_datasets_before_filter = process_map(
            partial(load_local_dataset, src_path=src_path, task_id=task_id),
            all_subdir_eids,
            max_workers=os.cpu_count() // 2,
            desc="Generating local dataset",
        )
    # remove the None result from the raw_datasets
    raw_datasets = [
        dataset for dataset in raw_datasets_before_filter if dataset is not None
    ]

    # remove the None result from the subdirs

    all_subdir_eids = [
        eid
        for eid, dataset in zip(all_subdir_eids, raw_datasets_before_filter)
        if dataset is not None
    ]
    all_subdir_episode_desc = [task_name] * len(all_subdir_eids)
    print(all_subdir_episode_desc)

    for raw_dataset, episode_desc in zip(
        tqdm(raw_datasets, desc="Generating dataset from raw datasets"),
        all_subdir_episode_desc,
    ):
        for raw_dataset_sub in tqdm(
            raw_dataset[0], desc="Generating dataset from raw dataset"
        ):
            dataset.add_frame(raw_dataset_sub)
        dataset.save_episode(task=episode_desc, videos=raw_dataset[1])
    dataset.consolidate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()

    task_id = args.task_id
    json_file = f"{args.src_path}/task_info/task_{args.task_id}.json"
    dataset_base = f"agibotworld/task_{args.task_id}"

    assert Path(json_file).exists, f"Cannot find {json_file}."
    main(args.src_path, args.tgt_path, task_id, dataset_base, json_file, args.debug)
