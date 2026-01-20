""" 
多任务并行分布式转换脚本
- 每个节点运行多个进程（默认8个）
- 每个进程处理一个完整的task
- 支持共享存储，task级别锁防止冲突
"""

import os
import json
import shutil
import logging
import argparse
import fcntl
import time
from pathlib import Path
from typing import Callable
from functools import partial
from math import ceil
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import torch
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
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

# 导入原脚本的常量和函数
import sys
sys.path.append(str(Path(__file__).parent))
from convert_to_lerobot import (
    FEATURES,
    AgiBotDataset,
    load_local_dataset,
    get_task_instruction,
    compute_stats,
)


class FileLock:
    """文件锁，防止多节点/多进程冲突"""
    
    def __init__(self, lockfile):
        self.lockfile = Path(lockfile)
        self.lockfile.parent.mkdir(parents=True, exist_ok=True)
        self.fd = None
    
    def acquire(self, timeout=30):
        """获取锁，超时返回False"""
        self.lockfile.touch()
        self.fd = open(self.lockfile, 'r')
        
        start_time = time.time()
        while True:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except IOError:
                if time.time() - start_time > timeout:
                    return False
                time.sleep(0.1)
    
    def release(self):
        """释放锁"""
        if self.fd:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
    
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Failed to acquire lock: {self.lockfile}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class TaskLockManager:
    """Task级别的锁管理"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.locks_dir = checkpoint_dir / "task_locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)
    
    def check_task_completed(self, task_id: str) -> bool:
        """检查task是否已完成"""
        checkpoint_file = self.checkpoint_dir / f"task_{task_id}.json"
        if not checkpoint_file.exists():
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return data.get('status') == 'completed'
        except Exception:
            return False
    
    def acquire_task_lock(self, task_id: str) -> FileLock | None:
        """尝试获取task锁"""
        lock_file = self.locks_dir / f"task_{task_id}.lock"
        lock = FileLock(lock_file)
        
        # 尝试获取锁，超时10秒
        if lock.acquire(timeout=10):
            # 再次检查是否已完成（防止竞态）
            if self.check_task_completed(task_id):
                lock.release()
                return None
            return lock
        else:
            return None
    
    def save_task_checkpoint(self, task_id: str, status: str, **extra_info):
        """保存task checkpoint（原子写入）"""
        checkpoint_file = self.checkpoint_dir / f"task_{task_id}.json"
        checkpoint_lock = self.checkpoint_dir / f".task_{task_id}.json.lock"
        
        data = {
            'task_id': task_id,
            'status': status,
            'timestamp': time.time(),
            **extra_info
        }
        
        # 原子写入
        with FileLock(checkpoint_lock):
            temp_file = checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(checkpoint_file)


def process_single_task(
    task_id: str,
    src_path: str,
    tgt_path: str,
    checkpoint_dir: str,
    num_workers_per_task: int = 4,
) -> dict:
    """处理单个task的所有episodes"""
    
    lock_manager = TaskLockManager(Path(checkpoint_dir))
    
    # 检查是否已完成
    if lock_manager.check_task_completed(task_id):
        return {
            'task_id': task_id,
            'status': 'skipped',
            'reason': 'already_completed'
        }
    
    # 尝试获取task锁
    task_lock = lock_manager.acquire_task_lock(task_id)
    if task_lock is None:
        return {
            'task_id': task_id,
            'status': 'skipped',
            'reason': 'locked_by_another_process'
        }
    
    try:
        # 保存开始状态
        lock_manager.save_task_checkpoint(
            task_id, 
            status='processing',
            pid=os.getpid()
        )
        
        # 获取task信息
        task_json = Path(src_path) / f"task_info/task_{task_id}.json"
        if not task_json.exists():
            raise FileNotFoundError(f"Task info not found: {task_json}")
        
        task_name = get_task_instruction(str(task_json))
        repo_id = f"agibotworld/task_{task_id}"
        dataset_path = Path(tgt_path) / repo_id
        
        # 如果目录已存在，说明可能：
        # 1. 正在被其他进程处理（不应该发生，因为有锁）
        # 2. 已完成（不应该发生，因为checkpoint检查会跳过）
        # 3. 之前失败/中断留下的（需要手动清理）
        # 安全起见，直接跳过让用户手动处理
        if dataset_path.exists():
            lock_manager.save_task_checkpoint(
                task_id,
                status='skipped',
                reason='directory_already_exists',
                pid=os.getpid()
            )
            return {
                'task_id': task_id,
                'status': 'skipped',
                'reason': 'directory_already_exists'
            }
        
        # 创建dataset
        dataset = AgiBotDataset.create(
            repo_id=repo_id,
            root=str(dataset_path),
            fps=30,
            robot_type="a2d",
            features=FEATURES,
        )
        
        # 获取所有episodes
        obs_dir = Path(src_path) / f"observations/{task_id}"
        if not obs_dir.exists():
            raise FileNotFoundError(f"Observations dir not found: {obs_dir}")
        
        all_episode_dirs = sorted([f for f in obs_dir.iterdir() if f.is_dir()])
        all_episode_ids = [int(f.name) for f in all_episode_dirs]
        
        if not all_episode_ids:
            raise ValueError(f"No episodes found for task {task_id}")
        
        # 使用线程池并行加载episodes，边加载边保存（避免内存累积）
        print(f"[Task {task_id}] Processing {len(all_episode_ids)} episodes...")
        processed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers_per_task) as executor:
            futures = {
                executor.submit(load_local_dataset, ep_id, src_path, int(task_id)): ep_id 
                for ep_id in all_episode_ids
            }
            
            with tqdm(total=len(all_episode_ids), desc=f"Task {task_id} - Processing episodes") as pbar:
                for future in as_completed(futures):
                    ep_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            # 立即保存，释放内存
                            frames, videos = result
                            try:
                                for frame in frames:
                                    dataset.add_frame(frame)
                                dataset.save_episode(task=task_name, videos=videos)
                                processed_count += 1
                            except Exception as save_error:
                                # 保存时出错
                                import traceback
                                print(f"\n[Task {task_id}] Episode {ep_id} save failed:")
                                print(f"  Error: {type(save_error).__name__}: {save_error}")
                                print(f"  Frames count: {len(frames)}")
                                traceback.print_exc()
                                failed_count += 1
                            finally:
                                # 主动释放内存
                                del result, frames, videos
                        else:
                            # load_local_dataset返回None表示跳过
                            failed_count += 1
                    except Exception as e:
                        # load_local_dataset加载时出错
                        import traceback
                        print(f"\n[Task {task_id}] Episode {ep_id} load failed:")
                        print(f"  Error: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        failed_count += 1
                    pbar.update(1)
        
        # Consolidate dataset
        print(f"[Task {task_id}] Consolidating dataset...")
        dataset.consolidate()
        
        # 保存完成状态
        lock_manager.save_task_checkpoint(
            task_id,
            status='completed',
            num_episodes=processed_count,
            failed_episodes=failed_count,
            pid=os.getpid()
        )
        
        return {
            'task_id': task_id,
            'status': 'completed',
            'num_episodes': processed_count,
            'failed_episodes': failed_count
        }
    
    except Exception as e:
        # 保存失败状态
        lock_manager.save_task_checkpoint(
            task_id,
            status='failed',
            error=str(e),
            pid=os.getpid()
        )
        
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(e)
        }
    
    finally:
        # 释放task锁
        task_lock.release()


def discover_all_tasks(src_path: str) -> list[str]:
    """自动发现所有可用的tasks"""
    task_info_dir = Path(src_path) / "task_info"
    if not task_info_dir.exists():
        return []
    
    task_files = list(task_info_dir.glob("task_*.json"))
    task_ids = []
    
    for task_file in task_files:
        # 提取task_id (task_123.json -> 123)
        task_id = task_file.stem.replace('task_', '')
        task_ids.append(task_id)
    
    return sorted(task_ids)


def main(
    src_path: str,
    tgt_path: str,
    checkpoint_dir: str,
    num_processes: int = 8,
    num_workers_per_task: int = 4,
    task_ids: list[str] = None,
):
    """
    主函数 - 竞争式处理，所有节点运行相同命令
    
    Args:
        src_path: 源数据路径
        tgt_path: 目标数据路径
        checkpoint_dir: checkpoint目录（必须是共享存储）
        num_processes: 每个节点的进程数（默认8）
        num_workers_per_task: 每个task内部的并行worker数
        task_ids: 指定要处理的task列表（None则自动发现）
    """
    
    # 自动发现tasks
    if task_ids is None:
        all_tasks = discover_all_tasks(src_path)
        print(f"Discovered {len(all_tasks)} tasks: {all_tasks}")
    else:
        all_tasks = task_ids
    
    if not all_tasks:
        print("No tasks found!")
        return
    
    hostname = os.uname().nodename
    pid = os.getpid()
    
    print(f"\n{'='*60}")
    print(f"Host: {hostname} (PID: {pid})")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Using {num_processes} processes")
    print(f"Mode: Competitive (all nodes process same task list)")
    print(f"{'='*60}\n")
    
    # 创建checkpoint目录
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 使用进程池并行处理多个tasks（竞争式）
    # 每个进程会尝试获取task锁，成功则处理，失败则跳过
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_single_task,
            [(task_id, src_path, tgt_path, checkpoint_dir, num_workers_per_task) 
             for task_id in all_tasks]
        )
    
    # 统计结果
    completed = sum(1 for r in results if r['status'] == 'completed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n{'='*60}")
    print(f"Host {hostname} finished!")
    print(f"Completed: {completed}, Skipped: {skipped}, Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task distributed converter (competitive mode)")
    parser.add_argument("--src_path", type=str, required=True, help="Source data path")
    parser.add_argument("--tgt_path", type=str, required=True, help="Target data path")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint directory (must be on shared storage)"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes per node (default: 8)"
    )
    parser.add_argument(
        "--num_workers_per_task",
        type=int,
        default=4,
        help="Number of workers for loading episodes within each task (default: 4)"
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        nargs='+',
        help="Specific task IDs to process (optional, auto-discover if not provided)"
    )
    
    args = parser.parse_args()
    
    main(
        src_path=args.src_path,
        tgt_path=args.tgt_path,
        checkpoint_dir=args.checkpoint_dir,
        num_processes=args.num_processes,
        num_workers_per_task=args.num_workers_per_task,
        task_ids=args.task_ids,
    )
