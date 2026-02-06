#!/usr/bin/env python3
"""
修复LeRobot数据集的问题

修复两类问题：
1. 相机数量不一致：删除相机数量不符合主流的 episode
2. Episode 索引不连续：重新编号 episode 使其连续
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict


class DataFixer:
    """数据修复器"""
    
    def __init__(self, output_path: Path, dry_run: bool = False):
        self.output_path = output_path
        self.dry_run = dry_run
        self.fixed_tasks = []
        
    def fix_all(self, task_ids: List[int] = None):
        """修复所有或指定的 tasks"""
        print("=" * 80)
        print("LeRobot 数据集问题修复")
        print("=" * 80)
        print()
        print(f"目录: {self.output_path}")
        print(f"模式: {'DRY RUN (只检查不修复)' if self.dry_run else '修复模式'}")
        print()
        
        # 扫描 tasks
        tasks = self._scan_tasks()
        
        if task_ids:
            tasks = {tid: tdir for tid, tdir in tasks.items() if tid in task_ids}
        
        if not tasks:
            print("❌ 没有找到需要修复的 tasks")
            return
        
        print(f"发现 {len(tasks)} 个 tasks")
        print()
        
        # 修复每个 task
        for task_id in sorted(tasks.keys()):
            task_dir = tasks[task_id]
            self._fix_task(task_id, task_dir)
        
        # 汇总
        print()
        print("=" * 80)
        print("修复完成")
        print("=" * 80)
        print()
        print(f"修复了 {len(self.fixed_tasks)} 个 tasks")
        
        if self.dry_run:
            print()
            print("⚠️  这是 DRY RUN 模式，没有实际修改任何文件")
            print("要执行修复，请去掉 --dry-run 参数")
    
    def _scan_tasks(self) -> Dict[int, Path]:
        """扫描所有 task 目录"""
        tasks = {}
        for task_dir in sorted(self.output_path.glob("task_*")):
            if not task_dir.is_dir():
                continue
            
            try:
                task_id = int(task_dir.name.replace("task_", ""))
                tasks[task_id] = task_dir
            except ValueError:
                continue
        
        return tasks
    
    def _fix_task(self, task_id: int, task_dir: Path):
        """修复单个 task"""
        print(f"检查 Task {task_id}...")
        
        # 1. 检查相机数量问题
        camera_issue = self._check_camera_consistency(task_id, task_dir)
        
        # 2. 检查 episode 连续性问题
        continuity_issue = self._check_episode_continuity(task_id, task_dir)
        
        # 如果没有问题，跳过
        if not camera_issue and not continuity_issue:
            print(f"  ✓ Task {task_id}: 没有问题")
            return
        
        # 有问题，需要修复
        print(f"  ⚠️  Task {task_id}: 发现问题")
        
        fixed = False
        
        # 修复相机数量问题
        if camera_issue:
            if self._fix_camera_issue(task_id, task_dir, camera_issue):
                fixed = True
        
        # 修复 episode 连续性问题
        if continuity_issue:
            if self._fix_continuity_issue(task_id, task_dir, continuity_issue):
                fixed = True
        
        if fixed:
            self.fixed_tasks.append(task_id)
            
            # 更新 meta/info.json
            if not self.dry_run:
                self._update_meta_info(task_id, task_dir)
    
    def _check_camera_consistency(self, task_id: int, task_dir: Path) -> Dict:
        """检查相机数量一致性"""
        videos_dir = task_dir / "videos"
        
        # 统计每个 episode 的相机数量
        episode_cameras = defaultdict(set)
        
        for video_file in videos_dir.rglob("episode_*.mp4"):
            try:
                episode_id = int(video_file.stem.replace("episode_", ""))
                camera_name = video_file.parent.name
                episode_cameras[episode_id].add(camera_name)
            except (ValueError, IndexError):
                continue
        
        if not episode_cameras:
            return None
        
        # 统计相机数量分布
        camera_counts = Counter(len(cameras) for cameras in episode_cameras.values())
        
        # 如果只有一种相机数量，没有问题
        if len(camera_counts) == 1:
            return None
        
        # 找出主流的相机数量（出现次数最多的）
        majority_count = camera_counts.most_common(1)[0][0]
        
        # 找出相机数量不一致的 episodes
        inconsistent_episodes = [
            ep_id for ep_id, cameras in episode_cameras.items()
            if len(cameras) != majority_count
        ]
        
        return {
            "camera_counts": dict(camera_counts),
            "majority_count": majority_count,
            "inconsistent_episodes": sorted(inconsistent_episodes),
            "episode_cameras": dict(episode_cameras)
        }
    
    def _check_episode_continuity(self, task_id: int, task_dir: Path) -> Dict:
        """检查 episode 连续性"""
        data_dir = task_dir / "data"
        
        # 获取所有 episode IDs
        episode_ids = set()
        for parquet_file in data_dir.rglob("episode_*.parquet"):
            try:
                episode_id = int(parquet_file.stem.replace("episode_", ""))
                episode_ids.add(episode_id)
            except ValueError:
                continue
        
        if not episode_ids:
            return None
        
        episode_ids = sorted(episode_ids)
        
        # 检查是否连续
        expected = list(range(len(episode_ids)))
        
        if episode_ids == expected:
            return None
        
        # 找出缺失的 episode
        missing = []
        for i, actual_id in enumerate(episode_ids):
            if actual_id != i:
                missing.append(i)
        
        return {
            "actual_episodes": episode_ids,
            "expected_episodes": expected,
            "needs_renumbering": True
        }
    
    def _fix_camera_issue(self, task_id: int, task_dir: Path, issue: Dict) -> bool:
        """修复相机数量问题"""
        majority_count = issue["majority_count"]
        inconsistent = issue["inconsistent_episodes"]
        camera_counts = issue["camera_counts"]
        
        print(f"      相机数量分布: {camera_counts}")
        print(f"      主流相机数量: {majority_count}")
        print(f"      不一致的 episodes: {len(inconsistent)} 个")
        
        if len(inconsistent) <= 5:
            print(f"        {inconsistent}")
        else:
            print(f"        {inconsistent[:5]} ... (共 {len(inconsistent)} 个)")
        
        # 删除相机数量不一致的 episodes
        print(f"      {'[DRY RUN] 将' if self.dry_run else ''}删除这些 episodes...")
        
        deleted_count = 0
        for episode_id in inconsistent:
            if self._delete_episode(task_id, task_dir, episode_id):
                deleted_count += 1
        
        if not self.dry_run:
            print(f"      ✓ 已删除 {deleted_count} 个 episodes")
        else:
            print(f"      [DRY RUN] 将删除 {deleted_count} 个 episodes")
        
        return deleted_count > 0
    
    def _fix_continuity_issue(self, task_id: int, task_dir: Path, issue: Dict) -> bool:
        """修复 episode 连续性问题"""
        actual = issue["actual_episodes"]
        expected = issue["expected_episodes"]
        
        print(f"      Episode 索引不连续")
        print(f"        实际: {len(actual)} 个 episodes, 索引范围 {actual[0]}~{actual[-1]}")
        print(f"        期望: {len(expected)} 个 episodes, 索引范围 {expected[0]}~{expected[-1]}")
        
        # 重新编号
        print(f"      {'[DRY RUN] 将' if self.dry_run else ''}重新编号 episodes...")
        
        if self._renumber_episodes(task_id, task_dir, actual):
            if not self.dry_run:
                print(f"      ✓ 已重新编号")
            else:
                print(f"      [DRY RUN] 将重新编号")
            return True
        
        return False
    
    def _delete_episode(self, task_id: int, task_dir: Path, episode_id: int) -> bool:
        """删除指定的 episode"""
        if self.dry_run:
            return True
        
        deleted = False
        
        # 删除 parquet 文件
        data_dir = task_dir / "data"
        for parquet_file in data_dir.rglob(f"episode_{episode_id:06d}.parquet"):
            try:
                parquet_file.unlink()
                deleted = True
            except Exception as e:
                print(f"        ⚠️  删除 parquet 失败: {e}")
        
        # 删除视频文件
        videos_dir = task_dir / "videos"
        for video_file in videos_dir.rglob(f"episode_{episode_id:06d}.mp4"):
            try:
                video_file.unlink()
                deleted = True
            except Exception as e:
                print(f"        ⚠️  删除视频失败: {e}")
        
        return deleted
    
    def _renumber_episodes(self, task_id: int, task_dir: Path, old_ids: List[int]) -> bool:
        """重新编号 episodes"""
        if self.dry_run:
            return True
        
        # 创建映射：old_id -> new_id
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        
        # 两阶段重命名（避免冲突）
        # 阶段1: 重命名为临时名称
        temp_files = []
        
        # 重命名 parquet 文件
        data_dir = task_dir / "data"
        for old_id, new_id in id_mapping.items():
            if old_id == new_id:
                continue
            
            for parquet_file in data_dir.rglob(f"episode_{old_id:06d}.parquet"):
                temp_file = parquet_file.with_name(f"episode_{old_id:06d}.parquet.tmp")
                try:
                    parquet_file.rename(temp_file)
                    temp_files.append((temp_file, new_id))
                except Exception as e:
                    print(f"        ⚠️  重命名 parquet 失败: {e}")
        
        # 重命名视频文件
        videos_dir = task_dir / "videos"
        for old_id, new_id in id_mapping.items():
            if old_id == new_id:
                continue
            
            for video_file in videos_dir.rglob(f"episode_{old_id:06d}.mp4"):
                temp_file = video_file.with_name(f"episode_{old_id:06d}.mp4.tmp")
                try:
                    video_file.rename(temp_file)
                    temp_files.append((temp_file, new_id))
                except Exception as e:
                    print(f"        ⚠️  重命名视频失败: {e}")
        
        # 阶段2: 从临时名称重命名为最终名称
        for temp_file, new_id in temp_files:
            if temp_file.suffix == ".tmp":
                # 去掉 .tmp 后缀，替换 episode ID
                original_name = temp_file.stem  # episode_XXXXXX.parquet 或 episode_XXXXXX.mp4
                extension = temp_file.stem.split('.')[-1]  # parquet 或 mp4
                final_name = f"episode_{new_id:06d}.{extension}"
                final_file = temp_file.with_name(final_name)
                
                try:
                    temp_file.rename(final_file)
                except Exception as e:
                    print(f"        ⚠️  最终重命名失败: {e}")
        
        return True
    
    def _update_meta_info(self, task_id: int, task_dir: Path):
        """更新 meta/info.json"""
        meta_info_file = task_dir / "meta" / "info.json"
        
        if not meta_info_file.exists():
            return
        
        try:
            # 读取现有信息
            with open(meta_info_file, 'r') as f:
                info = json.load(f)
            
            # 重新统计 episodes
            data_dir = task_dir / "data"
            episode_count = len(list(data_dir.rglob("episode_*.parquet")))
            
            # 更新
            info["total_episodes"] = episode_count
            
            # 写回
            with open(meta_info_file, 'w') as f:
                json.dump(info, f, indent=2)
            
            print(f"      ✓ 已更新 meta/info.json: total_episodes={episode_count}")
        
        except Exception as e:
            print(f"      ⚠️  更新 meta/info.json 失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='修复LeRobot数据集的问题',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查所有 tasks（不修复）
  python fix_data_issues.py --output-path output --dry-run
  
  # 修复所有 tasks
  python fix_data_issues.py --output-path output
  
  # 只修复指定的 tasks
  python fix_data_issues.py --output-path output --tasks 521 522 525 527 528
        """
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--tasks',
        type=int,
        nargs='+',
        help='指定要修复的 task IDs（不指定则修复所有）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只检查不修复'
    )
    
    args = parser.parse_args()
    
    if not args.output_path.exists():
        print(f"❌ 输出目录不存在: {args.output_path}")
        sys.exit(1)
    
    # 执行修复
    fixer = DataFixer(args.output_path, dry_run=args.dry_run)
    fixer.fix_all(task_ids=args.tasks)


if __name__ == '__main__':
    main()
