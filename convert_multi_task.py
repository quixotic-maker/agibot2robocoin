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

import h5py
import torch
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
)

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
        
        # 创建dataset
        dataset = AgiBotDataset.create(
            repo_id=repo_id,
            root=f"{tgt_path}/{repo_id}",
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
        
        # 并行加载所有episodes的数据
        print(f"[Task {task_id}] Loading {len(all_episode_ids)} episodes...")
        raw_datasets = process_map(
            partial(load_local_dataset, src_path=src_path, task_id=int(task_id)),
            all_episode_ids,
            max_workers=num_workers_per_task,
            desc=f"Task {task_id} - Loading episodes",
        )
        
        # 过滤掉None
        valid_datasets = [d for d in raw_datasets if d is not None]
        
        # 保存episodes到dataset
        print(f"[Task {task_id}] Saving {len(valid_datasets)} episodes to LeRobot format...")
        for raw_dataset in tqdm(valid_datasets, desc=f"Task {task_id} - Saving episodes"):
            frames, videos = raw_dataset
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode(task=task_name, videos=videos)
        
        # Consolidate dataset
        print(f"[Task {task_id}] Consolidating dataset...")
        dataset.consolidate()
        
        # 保存完成状态
        lock_manager.save_task_checkpoint(
            task_id,
            status='completed',
            num_episodes=len(valid_datasets),
            pid=os.getpid()
        )
        
        return {
            'task_id': task_id,
            'status': 'completed',
            'num_episodes': len(valid_datasets)
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
