#!/usr/bin/env python3
"""
检查LeRobot数据集的完整性和连续性

检查项：
1. Task 层面：列出所有 task（task 可以不连续）
2. Episode 层面：检查每个 task 内的 episode 是否连续
3. 数据完整性：检查文件是否完整
4. Episode 索引：检查 parquet 和视频文件的 episode 索引是否一致
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class DataIntegrityChecker:
    """数据完整性检查器"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.issues = []  # 存储发现的问题
        
    def check_all(self) -> Dict:
        """执行所有检查"""
        print("=" * 80)
        print("LeRobot 数据集完整性检查")
        print("=" * 80)
        print()
        print(f"检查目录: {self.output_path}")
        print()
        
        # 1. 扫描所有 tasks
        tasks = self._scan_tasks()
        print(f"✓ 发现 {len(tasks)} 个 tasks")
        
        if not tasks:
            print("\n❌ 没有发现任何 task 目录")
            return {"status": "error", "message": "No tasks found"}
        
        # 显示 task 范围
        task_ids = sorted(tasks.keys())
        print(f"  Task ID 范围: {task_ids[0]} ~ {task_ids[-1]}")
        
        # 检查 task 是否有间断（这是允许的，只是提示）
        missing_tasks = self._check_task_gaps(task_ids)
        if missing_tasks:
            print(f"  ℹ️  Task 有间断（这是正常的）: 缺少 {len(missing_tasks)} 个 tasks")
        
        print()
        
        # 2. 检查每个 task 的完整性
        print("=" * 80)
        print("检查每个 Task 的完整性")
        print("=" * 80)
        print()
        
        task_results = {}
        for task_id in task_ids:
            task_dir = tasks[task_id]
            result = self._check_task(task_id, task_dir)
            task_results[task_id] = result
        
        # 3. 汇总结果
        print()
        print("=" * 80)
        print("检查结果汇总")
        print("=" * 80)
        print()
        
        summary = self._generate_summary(task_results)
        self._print_summary(summary)
        
        return {
            "status": "success",
            "summary": summary,
            "tasks": task_results,
            "issues": self.issues
        }
    
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
                self.issues.append({
                    "type": "invalid_task_name",
                    "path": str(task_dir),
                    "message": f"无效的 task 目录名: {task_dir.name}"
                })
        
        return tasks
    
    def _check_task_gaps(self, task_ids: List[int]) -> List[int]:
        """检查 task ID 是否有间断"""
        if not task_ids:
            return []
        
        min_id = min(task_ids)
        max_id = max(task_ids)
        expected = set(range(min_id, max_id + 1))
        actual = set(task_ids)
        
        return sorted(expected - actual)
    
    def _check_task(self, task_id: int, task_dir: Path) -> Dict:
        """检查单个 task 的完整性"""
        print(f"检查 Task {task_id}...")
        
        result = {
            "task_id": task_id,
            "path": str(task_dir),
            "status": "ok",
            "issues": []
        }
        
        # 1. 检查基本目录结构
        required_dirs = ["meta", "data", "videos"]
        for dir_name in required_dirs:
            dir_path = task_dir / dir_name
            if not dir_path.exists():
                result["issues"].append(f"缺少 {dir_name}/ 目录")
                result["status"] = "error"
        
        if result["status"] == "error":
            print(f"  ❌ Task {task_id}: 目录结构不完整")
            for issue in result["issues"]:
                print(f"      - {issue}")
            self.issues.append(result)
            return result
        
        # 2. 检查 meta 文件
        meta_issues = self._check_meta_files(task_id, task_dir)
        if meta_issues:
            result["issues"].extend(meta_issues)
            result["status"] = "error"
        
        # 3. 读取 meta/info.json 获取 episode 信息
        meta_info = task_dir / "meta" / "info.json"
        if meta_info.exists():
            try:
                with open(meta_info, 'r') as f:
                    info = json.load(f)
                    result["total_episodes"] = info.get("total_episodes", 0)
                    result["total_frames"] = info.get("total_frames", 0)
            except Exception as e:
                result["issues"].append(f"无法读取 meta/info.json: {e}")
                result["status"] = "error"
        
        # 4. 检查 episode 连续性
        episode_issues = self._check_episodes(task_id, task_dir)
        if episode_issues:
            result["issues"].extend(episode_issues)
            if result["status"] == "ok":
                result["status"] = "warning"
        
        # 5. 检查数据文件
        data_issues = self._check_data_files(task_id, task_dir)
        if data_issues:
            result["issues"].extend(data_issues)
            if result["status"] == "ok":
                result["status"] = "warning"
        
        # 6. 检查视频文件
        video_issues = self._check_video_files(task_id, task_dir)
        if video_issues:
            result["issues"].extend(video_issues)
            if result["status"] == "ok":
                result["status"] = "warning"
        
        # 打印结果
        if result["status"] == "ok":
            print(f"  ✓ Task {task_id}: 完整 ({result.get('total_episodes', 0)} episodes)")
        elif result["status"] == "warning":
            print(f"  ⚠️  Task {task_id}: 有警告")
            for issue in result["issues"]:
                print(f"      - {issue}")
        else:
            print(f"  ❌ Task {task_id}: 有错误")
            for issue in result["issues"]:
                print(f"      - {issue}")
        
        if result["status"] != "ok":
            self.issues.append(result)
        
        return result
    
    def _check_meta_files(self, task_id: int, task_dir: Path) -> List[str]:
        """检查 meta 文件"""
        issues = []
        
        required_files = ["info.json", "episodes.jsonl", "tasks.jsonl"]
        for file_name in required_files:
            file_path = task_dir / "meta" / file_name
            if not file_path.exists():
                issues.append(f"缺少 meta/{file_name}")
        
        return issues
    
    def _check_episodes(self, task_id: int, task_dir: Path) -> List[str]:
        """检查 episode 连续性"""
        issues = []
        
        # 从 parquet 文件中提取 episode IDs
        data_dir = task_dir / "data"
        parquet_episodes = set()
        
        for parquet_file in data_dir.rglob("episode_*.parquet"):
            try:
                # 文件名格式: episode_000000.parquet
                episode_id = int(parquet_file.stem.replace("episode_", ""))
                parquet_episodes.add(episode_id)
            except ValueError:
                issues.append(f"无效的 parquet 文件名: {parquet_file.name}")
        
        # 从视频文件中提取 episode IDs
        videos_dir = task_dir / "videos"
        video_episodes = set()
        
        for video_file in videos_dir.rglob("episode_*.mp4"):
            try:
                episode_id = int(video_file.stem.replace("episode_", ""))
                video_episodes.add(episode_id)
            except ValueError:
                issues.append(f"无效的视频文件名: {video_file.name}")
        
        if not parquet_episodes:
            issues.append("没有找到任何 parquet 文件")
            return issues
        
        if not video_episodes:
            issues.append("没有找到任何视频文件")
            return issues
        
        # 检查 parquet 和 video 的 episode 是否一致
        if parquet_episodes != video_episodes:
            only_parquet = parquet_episodes - video_episodes
            only_video = video_episodes - parquet_episodes
            
            if only_parquet:
                issues.append(f"只有 parquet 没有视频的 episodes: {sorted(only_parquet)}")
            if only_video:
                issues.append(f"只有视频没有 parquet 的 episodes: {sorted(only_video)}")
        
        # 检查 episode 连续性
        all_episodes = sorted(parquet_episodes & video_episodes)
        if all_episodes:
            expected = list(range(len(all_episodes)))
            if all_episodes != expected:
                issues.append(f"Episode 索引不连续: 期望 {expected}, 实际 {all_episodes}")
        
        return issues
    
    def _check_data_files(self, task_id: int, task_dir: Path) -> List[str]:
        """检查数据文件"""
        issues = []
        
        data_dir = task_dir / "data"
        parquet_files = list(data_dir.rglob("*.parquet"))
        
        if not parquet_files:
            issues.append("data/ 目录下没有 parquet 文件")
        
        return issues
    
    def _check_video_files(self, task_id: int, task_dir: Path) -> List[str]:
        """检查视频文件"""
        issues = []
        
        videos_dir = task_dir / "videos"
        video_files = list(videos_dir.rglob("*.mp4"))
        
        if not video_files:
            issues.append("videos/ 目录下没有视频文件")
        
        # 检查相机数量是否一致
        episodes_cameras = defaultdict(set)
        for video_file in video_files:
            try:
                # 路径格式: videos/chunk-000/observation.images.camera1/episode_000000.mp4
                episode_id = int(video_file.stem.replace("episode_", ""))
                camera_name = video_file.parent.name  # observation.images.camera1
                episodes_cameras[episode_id].add(camera_name)
            except (ValueError, IndexError):
                continue
        
        # 检查每个 episode 的相机数量是否一致
        if episodes_cameras:
            camera_counts = [len(cameras) for cameras in episodes_cameras.values()]
            if len(set(camera_counts)) > 1:
                issues.append(f"不同 episode 的相机数量不一致: {set(camera_counts)}")
        
        return issues
    
    def _generate_summary(self, task_results: Dict) -> Dict:
        """生成汇总信息"""
        summary = {
            "total_tasks": len(task_results),
            "ok_tasks": 0,
            "warning_tasks": 0,
            "error_tasks": 0,
            "total_episodes": 0,
            "total_frames": 0
        }
        
        for result in task_results.values():
            if result["status"] == "ok":
                summary["ok_tasks"] += 1
            elif result["status"] == "warning":
                summary["warning_tasks"] += 1
            else:
                summary["error_tasks"] += 1
            
            summary["total_episodes"] += result.get("total_episodes", 0)
            summary["total_frames"] += result.get("total_frames", 0)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """打印汇总信息"""
        print(f"总 Tasks: {summary['total_tasks']}")
        print(f"  ✓ 完整: {summary['ok_tasks']}")
        print(f"  ⚠️  警告: {summary['warning_tasks']}")
        print(f"  ❌ 错误: {summary['error_tasks']}")
        print()
        print(f"总 Episodes: {summary['total_episodes']}")
        print(f"总 Frames: {summary['total_frames']}")
        print()
        
        if summary['error_tasks'] > 0 or summary['warning_tasks'] > 0:
            print("⚠️  发现问题，请查看上面的详细信息")
        else:
            print("✅ 所有数据完整！")


def main():
    parser = argparse.ArgumentParser(
        description='检查LeRobot数据集的完整性和连续性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查输出目录
  python check_data_integrity.py --output-path output
  
  # 保存检查结果到文件
  python check_data_integrity.py --output-path output --save-report report.json
        """
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        required=True,
        help='输出目录路径（包含 task_xxx 的目录）'
    )
    
    parser.add_argument(
        '--save-report',
        type=Path,
        help='保存检查报告到 JSON 文件'
    )
    
    args = parser.parse_args()
    
    if not args.output_path.exists():
        print(f"❌ 输出目录不存在: {args.output_path}")
        sys.exit(1)
    
    # 执行检查
    checker = DataIntegrityChecker(args.output_path)
    result = checker.check_all()
    
    # 保存报告
    if args.save_report:
        with open(args.save_report, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print()
        print(f"✓ 检查报告已保存到: {args.save_report}")
    
    # 返回退出码
    if result["summary"]["error_tasks"] > 0:
        sys.exit(1)
    elif result["summary"]["warning_tasks"] > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
