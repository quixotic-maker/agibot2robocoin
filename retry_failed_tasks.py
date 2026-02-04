#!/usr/bin/env python3
"""
重新运行失败的tasks

识别失败或不完整的tasks，清理它们，然后重新运行转换器
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def check_task_completeness(task_dir: Path) -> Tuple[bool, str]:
    """
    检查task输出是否完整
    
    Returns:
        (is_complete, reason)
    """
    # 检查meta/info.json
    meta_info = task_dir / "meta" / "info.json"
    if not meta_info.exists():
        return False, "缺少 meta/info.json"
    
    # 检查meta/episodes.jsonl
    meta_episodes = task_dir / "meta" / "episodes.jsonl"
    if not meta_episodes.exists():
        return False, "缺少 meta/episodes.jsonl"
    
    # 检查meta/tasks.jsonl
    meta_tasks = task_dir / "meta" / "tasks.jsonl"
    if not meta_tasks.exists():
        return False, "缺少 meta/tasks.jsonl"
    
    # 检查是否有数据文件
    data_dir = task_dir / "data"
    if not data_dir.exists():
        return False, "缺少 data 目录"
    
    # data目录结构: data/chunk-000/episode_000000.parquet
    parquet_files = list(data_dir.rglob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        # 尝试更宽松的搜索
        parquet_files = list(data_dir.rglob("*.parquet"))
    if not parquet_files:
        return False, "没有 parquet 文件"
    
    # 检查是否有视频文件
    videos_dir = task_dir / "videos"
    if not videos_dir.exists():
        return False, "缺少 videos 目录"
    
    # videos目录结构: videos/chunk-000/observation.images.camera1/episode_000000.mp4
    video_files = list(videos_dir.rglob("chunk-*/observation.images.*/episode_*.mp4"))
    if not video_files:
        # 尝试更宽松的搜索
        video_files = list(videos_dir.rglob("*.mp4"))
    if not video_files:
        return False, "没有视频文件"
    
    # 检查meta/info.json是否有效
    try:
        with open(meta_info, 'r') as f:
            info = json.load(f)
            
            # 检查必需的元信息字段
            required_fields = [
                'total_episodes',
                'total_frames', 
                'total_tasks',
                'fps',
                'features'
            ]
            
            missing_fields = [field for field in required_fields if field not in info]
            if missing_fields:
                return False, f"meta/info.json 缺少字段: {', '.join(missing_fields)}"
            
            # 检查 features 是否包含必需的特征
            features = info.get('features', {})
            required_features = ['observation.state', 'action']
            missing_features = [feat for feat in required_features if feat not in features]
            if missing_features:
                return False, f"meta/info.json features 缺少: {', '.join(missing_features)}"
            
            # 检查 total_episodes 是否大于0
            if info.get('total_episodes', 0) <= 0:
                return False, "meta/info.json total_episodes 必须大于0"
                
    except json.JSONDecodeError as e:
        return False, f"meta/info.json JSON格式错误: {e}"
    except Exception as e:
        return False, f"meta/info.json 读取失败: {e}"
    
    return True, "完整"


def find_failed_tasks(output_path: Path) -> List[Tuple[int, Path, str]]:
    """
    查找失败或不完整的tasks
    
    Returns:
        List of (task_id, task_dir, reason)
    """
    failed_tasks = []
    
    # 查找所有task目录
    task_dirs = sorted(output_path.glob("task_*"))
    
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        
        # 提取task_id
        try:
            task_id = int(task_dir.name.replace("task_", ""))
        except ValueError:
            continue
        
        # 检查完整性
        is_complete, reason = check_task_completeness(task_dir)
        
        if not is_complete:
            failed_tasks.append((task_id, task_dir, reason))
    
    return failed_tasks


def clean_failed_task(task_dir: Path, lock_dir: Path, task_id: int, dry_run: bool = False):
    """清理失败的task"""
    print(f"  清理 {task_dir.name}...")
    
    if dry_run:
        print(f"    [DRY RUN] 将删除: {task_dir}")
    else:
        # 删除task目录
        if task_dir.exists():
            shutil.rmtree(task_dir)
            print(f"    ✓ 已删除: {task_dir}")
    
    # 删除锁文件（如果存在）
    lock_file = lock_dir / f"task_{task_id}.lock"
    if lock_file.exists():
        if dry_run:
            print(f"    [DRY RUN] 将删除锁: {lock_file}")
        else:
            lock_file.unlink()
            print(f"    ✓ 已删除锁: {lock_file}")


def main():
    parser = argparse.ArgumentParser(
        description='识别并重新运行失败的tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查失败的tasks（不执行清理）
  python retry_failed_tasks.py --output-path output --dry-run
  
  # 清理失败的tasks
  python retry_failed_tasks.py --output-path output --clean
  
  # 清理并重新运行
  python retry_failed_tasks.py \\
    --output-path output \\
    --dataset-path data/agibot \\
    --repo-id agibot/dataset \\
    --clean --retry
        """
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=Path,
        help='数据集路径（用于重新运行）'
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        help='Repository ID（用于重新运行）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只检查，不执行清理'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='清理失败的tasks'
    )
    
    parser.add_argument(
        '--retry',
        action='store_true',
        help='清理后重新运行转换器'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Worker数量（用于重新运行）'
    )
    
    args = parser.parse_args()
    
    if not args.output_path.exists():
        print(f"❌ 输出目录不存在: {args.output_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("检查失败的Tasks")
    print("=" * 80)
    print()
    
    # 查找失败的tasks
    print(f"扫描输出目录: {args.output_path}")
    failed_tasks = find_failed_tasks(args.output_path)
    
    if not failed_tasks:
        print("\n✅ 没有发现失败的tasks，所有tasks都完整！")
        return
    
    print(f"\n发现 {len(failed_tasks)} 个失败或不完整的tasks:")
    print()
    
    for task_id, task_dir, reason in failed_tasks:
        print(f"  Task {task_id}: {reason}")
        print(f"    路径: {task_dir}")
    
    print()
    
    # 如果只是dry-run，到此结束
    if args.dry_run and not args.clean:
        print("=" * 80)
        print("Dry Run 模式 - 未执行任何操作")
        print("=" * 80)
        print()
        print("要清理这些tasks，请运行:")
        print(f"  python retry_failed_tasks.py --output-path {args.output_path} --clean")
        return
    
    # 清理失败的tasks
    if args.clean or args.retry:
        print("=" * 80)
        print("清理失败的Tasks")
        print("=" * 80)
        print()
        
        lock_dir = args.output_path / ".locks"
        
        for task_id, task_dir, reason in failed_tasks:
            clean_failed_task(task_dir, lock_dir, task_id, dry_run=args.dry_run)
        
        if not args.dry_run:
            print()
            print(f"✓ 已清理 {len(failed_tasks)} 个失败的tasks")
    
    # 重新运行转换器
    if args.retry and not args.dry_run:
        if not args.dataset_path or not args.repo_id:
            print()
            print("❌ 要重新运行转换器，需要提供 --dataset-path 和 --repo-id")
            sys.exit(1)
        
        print()
        print("=" * 80)
        print("重新运行转换器")
        print("=" * 80)
        print()
        
        # 构建命令
        cmd = [
            'python',
            'agibot_distributed_converter.py',
            '--dataset-path', str(args.dataset_path),
            '--output-path', str(args.output_path),
            '--repo-id', args.repo_id
        ]
        
        if args.max_workers:
            cmd.extend(['--max-workers', str(args.max_workers)])
        
        print("执行命令:")
        print(' '.join(cmd))
        print()
        
        # 运行转换器
        try:
            subprocess.run(cmd, check=True)
            print()
            print("✅ 转换完成")
        except subprocess.CalledProcessError as e:
            print()
            print(f"❌ 转换失败: {e}")
            sys.exit(1)
    
    print()


if __name__ == '__main__':
    main()
