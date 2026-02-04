#!/usr/bin/env python3
"""
增量修复不完整的tasks

检测每个task缺少什么，只补充缺少的部分（而不是删除重来）
这样可以节省时间和资源，避免重新处理已经完成的部分。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class TaskStatus:
    """Task的完整性状态 - 对应11项检查"""
    task_id: int
    task_dir: Path
    is_complete: bool
    
    # 11项检查结果
    check_1_info_json_exists: bool          # meta/info.json 存在
    check_2_episodes_jsonl_exists: bool     # meta/episodes.jsonl 存在
    check_3_tasks_jsonl_exists: bool        # meta/tasks.jsonl 存在
    check_4_data_dir_exists: bool           # data/ 目录存在
    check_5_has_parquet_files: bool         # data/ 下有 .parquet 文件
    check_6_videos_dir_exists: bool         # videos/ 目录存在
    check_7_has_video_files: bool           # videos/ 下有 .mp4 文件
    check_8_info_json_readable: bool        # meta/info.json 可读取（有效JSON）
    check_9_info_json_has_required_fields: bool  # meta/info.json 包含必需字段
    check_10_features_complete: bool        # features 包含必需特征
    check_11_total_episodes_valid: bool     # total_episodes > 0
    
    # 详细信息
    missing_items: List[str]                # 缺失项列表（用于显示）
    info_json_issues: List[str]             # info.json的具体问题
    
    # 辅助字段（用于分类）
    has_meta_dir: bool
    has_data_dir: bool
    has_videos_dir: bool


def check_task_status(task_dir: Path) -> TaskStatus:
    """
    详细检查task的状态 - 执行完整的11项检查
    
    Returns:
        TaskStatus对象，包含11项检查的详细结果
    """
    task_id = int(task_dir.name.replace("task_", ""))
    missing_items = []
    info_json_issues = []
    
    # 检查目录
    meta_dir = task_dir / "meta"
    data_dir = task_dir / "data"
    videos_dir = task_dir / "videos"
    
    has_meta_dir = meta_dir.exists()
    has_data_dir = data_dir.exists()
    has_videos_dir = videos_dir.exists()
    
    # ========== 11项检查 ==========
    
    # 检查1: meta/info.json 存在
    meta_info = meta_dir / "info.json" if has_meta_dir else None
    check_1 = meta_info.exists() if meta_info else False
    if not check_1:
        missing_items.append("meta/info.json")
    
    # 检查2: meta/episodes.jsonl 存在
    meta_episodes = meta_dir / "episodes.jsonl" if has_meta_dir else None
    check_2 = meta_episodes.exists() if meta_episodes else False
    if not check_2:
        missing_items.append("meta/episodes.jsonl")
    
    # 检查3: meta/tasks.jsonl 存在
    meta_tasks = meta_dir / "tasks.jsonl" if has_meta_dir else None
    check_3 = meta_tasks.exists() if meta_tasks else False
    if not check_3:
        missing_items.append("meta/tasks.jsonl")
    
    # 检查4: data/ 目录存在
    check_4 = has_data_dir
    if not check_4:
        missing_items.append("data目录")
    
    # 检查5: data/ 下有 .parquet 文件（在chunk-xxx子目录中）
    check_5 = False
    if check_4:
        # data目录结构: data/chunk-000/episode_000000.parquet
        parquet_files = list(data_dir.rglob("chunk-*/episode_*.parquet"))
        check_5 = len(parquet_files) > 0
        
        # 如果没找到，尝试更宽松的搜索
        if not check_5:
            parquet_files = list(data_dir.rglob("*.parquet"))
            check_5 = len(parquet_files) > 0
    if not check_5:
        missing_items.append("parquet文件")
    
    # 检查6: videos/ 目录存在
    check_6 = has_videos_dir
    if not check_6:
        missing_items.append("videos目录")
    
    # 检查7: videos/ 下有 .mp4 文件（在chunk-xxx/observation.images.xxx子目录中）
    check_7 = False
    if check_6:
        # videos目录结构: videos/chunk-000/observation.images.camera1/episode_000000.mp4
        video_files = list(videos_dir.rglob("chunk-*/observation.images.*/episode_*.mp4"))
        check_7 = len(video_files) > 0
        
        # 如果没找到，尝试更宽松的搜索
        if not check_7:
            video_files = list(videos_dir.rglob("*.mp4"))
            check_7 = len(video_files) > 0
    if not check_7:
        missing_items.append("视频文件")
    
    # 检查8: meta/info.json 可读取（有效JSON）
    check_8 = False
    info_data = None
    if check_1:
        try:
            with open(meta_info, 'r') as f:
                info_data = json.load(f)
            check_8 = True
        except json.JSONDecodeError as e:
            info_json_issues.append(f"JSON格式错误: {e}")
        except Exception as e:
            info_json_issues.append(f"读取失败: {e}")
    
    # 检查9: meta/info.json 包含必需字段
    check_9 = False
    if check_8 and info_data:
        required_fields = [
            'total_episodes',
            'total_frames',
            'total_tasks',
            'fps',
            'features'
        ]
        
        missing_fields = [field for field in required_fields if field not in info_data]
        if missing_fields:
            info_json_issues.append(f"缺少字段: {', '.join(missing_fields)}")
        else:
            check_9 = True
    
    # 检查10: features 包含必需特征
    check_10 = False
    if check_9 and info_data:
        features = info_data.get('features', {})
        required_features = ['observation.state', 'action']
        missing_features = [feat for feat in required_features if feat not in features]
        if missing_features:
            info_json_issues.append(f"features缺少: {', '.join(missing_features)}")
        else:
            check_10 = True
    
    # 检查11: total_episodes > 0
    check_11 = False
    if check_9 and info_data:
        total_episodes = info_data.get('total_episodes', 0)
        if total_episodes <= 0:
            info_json_issues.append("total_episodes必须大于0")
        else:
            check_11 = True
    
    # 判断是否完整（所有11项检查都通过）
    is_complete = all([
        check_1, check_2, check_3, check_4, check_5, check_6,
        check_7, check_8, check_9, check_10, check_11
    ])
    
    return TaskStatus(
        task_id=task_id,
        task_dir=task_dir,
        is_complete=is_complete,
        
        # 11项检查结果
        check_1_info_json_exists=check_1,
        check_2_episodes_jsonl_exists=check_2,
        check_3_tasks_jsonl_exists=check_3,
        check_4_data_dir_exists=check_4,
        check_5_has_parquet_files=check_5,
        check_6_videos_dir_exists=check_6,
        check_7_has_video_files=check_7,
        check_8_info_json_readable=check_8,
        check_9_info_json_has_required_fields=check_9,
        check_10_features_complete=check_10,
        check_11_total_episodes_valid=check_11,
        
        # 详细信息
        missing_items=missing_items,
        info_json_issues=info_json_issues,
        
        # 辅助字段
        has_meta_dir=has_meta_dir,
        has_data_dir=has_data_dir,
        has_videos_dir=has_videos_dir
    )


def find_incomplete_tasks(output_path: Path) -> List[TaskStatus]:
    """
    查找所有不完整的tasks
    
    Returns:
        List of TaskStatus objects
    """
    incomplete_tasks = []
    
    # 查找所有task目录
    task_dirs = sorted(output_path.glob("task_*"))
    
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        
        status = check_task_status(task_dir)
        
        if not status.is_complete:
            incomplete_tasks.append(status)
    
    return incomplete_tasks


def categorize_tasks(tasks: List[TaskStatus]) -> Dict[str, List[TaskStatus]]:
    """
    将tasks按照缺失类型分类
    
    基于11项检查结果进行智能分类
    
    Returns:
        Dict mapping category to list of tasks
    """
    categories = {
        'only_info_json_content': [],  # 只是info.json内容有问题（文件存在但内容不对）
        'only_meta_files': [],         # 只缺meta文件（数据和视频都有）
        'only_videos': [],             # 只缺视频（meta和数据都有）
        'only_parquet': [],            # 只缺parquet（meta和视频都有）
        'meta_and_videos': [],         # 缺meta和视频（有数据）
        'meta_and_parquet': [],        # 缺meta和parquet（有视频）
        'videos_and_parquet': [],      # 缺视频和parquet（有meta）
        'all_missing': [],             # 全部缺失或几乎全部缺失
    }
    
    for task in tasks:
        # 只是info.json内容有问题（检查8-11失败，但1-7都通过）
        if (task.check_1_info_json_exists and
            task.check_2_episodes_jsonl_exists and
            task.check_3_tasks_jsonl_exists and
            task.check_4_data_dir_exists and
            task.check_5_has_parquet_files and
            task.check_6_videos_dir_exists and
            task.check_7_has_video_files and
            not (task.check_8_info_json_readable and 
                 task.check_9_info_json_has_required_fields and
                 task.check_10_features_complete and
                 task.check_11_total_episodes_valid)):
            categories['only_info_json_content'].append(task)
        
        # 只缺meta文件（数据和视频都有）
        elif (task.check_5_has_parquet_files and
              task.check_7_has_video_files and
              not (task.check_1_info_json_exists and
                   task.check_2_episodes_jsonl_exists and
                   task.check_3_tasks_jsonl_exists)):
            categories['only_meta_files'].append(task)
        
        # 只缺视频（meta和数据都有）
        elif (task.check_1_info_json_exists and
              task.check_2_episodes_jsonl_exists and
              task.check_3_tasks_jsonl_exists and
              task.check_5_has_parquet_files and
              not task.check_7_has_video_files):
            categories['only_videos'].append(task)
        
        # 只缺parquet（meta和视频都有）
        elif (task.check_1_info_json_exists and
              task.check_2_episodes_jsonl_exists and
              task.check_3_tasks_jsonl_exists and
              task.check_7_has_video_files and
              not task.check_5_has_parquet_files):
            categories['only_parquet'].append(task)
        
        # 缺meta和视频（有数据）
        elif (task.check_5_has_parquet_files and
              not task.check_7_has_video_files and
              not (task.check_1_info_json_exists and
                   task.check_2_episodes_jsonl_exists and
                   task.check_3_tasks_jsonl_exists)):
            categories['meta_and_videos'].append(task)
        
        # 缺meta和parquet（有视频）
        elif (task.check_7_has_video_files and
              not task.check_5_has_parquet_files and
              not (task.check_1_info_json_exists and
                   task.check_2_episodes_jsonl_exists and
                   task.check_3_tasks_jsonl_exists)):
            categories['meta_and_parquet'].append(task)
        
        # 缺视频和parquet（有meta）
        elif (task.check_1_info_json_exists and
              task.check_2_episodes_jsonl_exists and
              task.check_3_tasks_jsonl_exists and
              not task.check_5_has_parquet_files and
              not task.check_7_has_video_files):
            categories['videos_and_parquet'].append(task)
        
        # 全部缺失或几乎全部缺失
        else:
            categories['all_missing'].append(task)
    
    return categories


def print_task_summary(tasks: List[TaskStatus], categories: Dict[str, List[TaskStatus]]):
    """打印任务摘要 - 显示所有缺失内容的详细列表"""
    print(f"\n发现 {len(tasks)} 个不完整的tasks\n")
    print("=" * 80)
    print("详细检查结果（11项检查）")
    print("=" * 80)
    print()
    
    for task in tasks:
        print(f"Task {task.task_id}: {task.task_dir}")
        print("-" * 80)
        
        # 显示所有11项检查的结果
        checks = [
            (1, "meta/info.json 存在", task.check_1_info_json_exists),
            (2, "meta/episodes.jsonl 存在", task.check_2_episodes_jsonl_exists),
            (3, "meta/tasks.jsonl 存在", task.check_3_tasks_jsonl_exists),
            (4, "data/ 目录存在", task.check_4_data_dir_exists),
            (5, "data/ 下有 parquet 文件", task.check_5_has_parquet_files),
            (6, "videos/ 目录存在", task.check_6_videos_dir_exists),
            (7, "videos/ 下有视频文件", task.check_7_has_video_files),
            (8, "info.json 可读取（有效JSON）", task.check_8_info_json_readable),
            (9, "info.json 包含必需字段", task.check_9_info_json_has_required_fields),
            (10, "features 包含必需特征", task.check_10_features_complete),
            (11, "total_episodes > 0", task.check_11_total_episodes_valid),
        ]
        
        # 分别显示通过和未通过的检查
        passed_checks = [(num, name) for num, name, passed in checks if passed]
        failed_checks = [(num, name) for num, name, passed in checks if not passed]
        
        if passed_checks:
            print("  ✅ 已通过的检查:")
            for num, name in passed_checks:
                print(f"     [{num}] {name}")
        
        if failed_checks:
            print("  ❌ 未通过的检查（缺失内容）:")
            for num, name in failed_checks:
                print(f"     [{num}] {name}")
        
        # 显示info.json的具体问题
        if task.info_json_issues:
            print("  ⚠️  info.json 具体问题:")
            for issue in task.info_json_issues:
                print(f"     - {issue}")
        
        print()
    
    # 按类别汇总
    print("=" * 80)
    print("按修复难度分类")
    print("=" * 80)
    print()
    
    category_names = {
        'only_info_json_content': '只需修复 meta/info.json 内容（最快）',
        'only_meta_files': '只需生成 meta 文件（很快）',
        'only_videos': '只需生成视频文件（中等）',
        'only_parquet': '只需生成 parquet 文件（中等）',
        'meta_and_videos': '需要生成 meta + 视频',
        'meta_and_parquet': '需要生成 meta + parquet',
        'videos_and_parquet': '需要生成视频 + parquet',
        'all_missing': '需要完全重新处理',
    }
    
    for cat_key, cat_name in category_names.items():
        cat_tasks = categories[cat_key]
        if cat_tasks:
            task_ids = [str(t.task_id) for t in cat_tasks]
            print(f"【{cat_name}】")
            print(f"  数量: {len(cat_tasks)} 个")
            print(f"  Tasks: {', '.join(task_ids)}")
            print()


def fix_info_json(task: TaskStatus, dataset_path: Path, dry_run: bool = False) -> bool:
    """
    修复info.json文件
    
    通过重新读取parquet和视频文件来生成正确的info.json
    """
    print(f"  修复 Task {task.task_id} 的 meta/info.json...")
    
    if dry_run:
        print(f"    [DRY RUN] 将重新生成 info.json")
        return True
    
    try:
        # 读取现有的episodes.jsonl获取episode信息
        episodes_file = task.task_dir / "meta" / "episodes.jsonl"
        if not episodes_file.exists():
            print(f"    ❌ 缺少 episodes.jsonl，无法修复")
            return False
        
        episodes_info = []
        with open(episodes_file, 'r') as f:
            for line in f:
                episodes_info.append(json.loads(line.strip()))
        
        if not episodes_info:
            print(f"    ❌ episodes.jsonl为空")
            return False
        
        # 读取第一个parquet文件获取维度信息
        first_episode_idx = episodes_info[0]["episode_index"]
        chunk_num = first_episode_idx // 1000
        chunk_name = f"chunk-{chunk_num:03d}"
        parquet_file = task.task_dir / "data" / chunk_name / f"episode_{first_episode_idx:06d}.parquet"
        
        if not parquet_file.exists():
            print(f"    ❌ 找不到parquet文件: {parquet_file}")
            return False
        
        import pandas as pd
        df = pd.read_parquet(parquet_file)
        
        # 获取维度
        state_dim = len(df['observation.state'].iloc[0]) if 'observation.state' in df.columns else 0
        action_dim = len(df['action'].iloc[0]) if 'action' in df.columns else 0
        
        # 获取相机列表
        camera_names = []
        videos_chunk_dir = task.task_dir / "videos" / chunk_name
        if videos_chunk_dir.exists():
            for camera_dir in videos_chunk_dir.iterdir():
                if camera_dir.is_dir() and camera_dir.name.startswith("observation.images."):
                    camera_name = camera_dir.name.replace("observation.images.", "")
                    camera_names.append(camera_name)
        
        # 计算总帧数
        total_frames = sum(ep['length'] for ep in episodes_info)
        total_episodes = len(episodes_info)
        
        # 构建info.json
        info = {
            "codebase_version": "v2.0",
            "robot_type": "agibot",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": total_episodes * len(camera_names),
            "total_chunks": len(set(ep['episode_index'] // 1000 for ep in episodes_info)),
            "chunks_size": 1000,
            "fps": 30,
            "splits": {
                "train": f"0:{total_episodes}"
            },
            "data_path": f"data/{chunk_name}",
            "video_path": f"videos/{chunk_name}",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [state_dim],
                    "names": [f"state_{i+1}" for i in range(state_dim)]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [action_dim],
                    "names": [f"action_{i+1}" for i in range(action_dim)]
                }
            }
        }
        
        # 添加相机特征
        for camera_name in sorted(camera_names):
            info["features"][f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": [3, 480, 640],
                "names": ["channel", "height", "width"]
            }
        
        # 写入info.json
        info_file = task.task_dir / "meta" / "info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"    ✓ 已修复 info.json")
        return True
        
    except Exception as e:
        print(f"    ❌ 修复失败: {e}")
        return False


def suggest_fix_strategy(categories: Dict[str, List[TaskStatus]]):
    """建议修复策略 - 基于11项检查结果"""
    print("\n" + "=" * 80)
    print("修复建议")
    print("=" * 80)
    print()
    
    # info.json内容问题 - 最简单，可以直接修复
    if categories['only_info_json_content']:
        print(f"✅ {len(categories['only_info_json_content'])} 个tasks只需修复info.json内容（最快）")
        print("   检查1-7都通过，只是info.json内容有问题")
        print("   运行: python fix_incomplete_tasks.py --output-path output --fix-info-json")
        print()
    
    # 只缺meta文件 - 很快，但需要重新生成
    if categories['only_meta_files']:
        print(f"⚡ {len(categories['only_meta_files'])} 个tasks只需生成meta文件（很快）")
        print("   数据和视频都有，只需重新生成meta文件")
        print("   建议: 删除这些tasks，重新运行转换器")
        print()
    
    # 只缺视频 - 中等耗时
    if categories['only_videos']:
        print(f"🎥 {len(categories['only_videos'])} 个tasks只需生成视频（中等耗时）")
        print("   parquet数据和meta都有，只需重新生成视频")
        print("   建议: 删除这些tasks，重新运行转换器")
        print()
    
    # 只缺parquet - 中等耗时
    if categories['only_parquet']:
        print(f"📊 {len(categories['only_parquet'])} 个tasks只需生成parquet（中等耗时）")
        print("   视频和meta都有，只需重新生成parquet数据")
        print("   建议: 删除这些tasks，重新运行转换器")
        print()
    
    # 需要完全重新处理
    total_need_full = (
        len(categories['meta_and_videos']) +
        len(categories['meta_and_parquet']) +
        len(categories['videos_and_parquet']) +
        len(categories['all_missing'])
    )
    
    if total_need_full > 0:
        print(f"🔄 {total_need_full} 个tasks需要完全重新处理")
        print("   建议: 删除这些tasks，重新运行转换器")
        print()
    
    print("通用修复命令:")
    print("  # 删除不完整的tasks")
    print("  python fix_incomplete_tasks.py --output-path output --clean")
    print()
    print("  # 重新运行转换器")
    print("  python agibot_distributed_converter.py \\")
    print("      --dataset-path data/agibot \\")
    print("      --output-path output \\")
    print("      --repo-id agibot/dataset")
    print()


def clean_tasks(tasks: List[TaskStatus], output_path: Path, dry_run: bool = False):
    """清理指定的tasks"""
    print("\n" + "=" * 80)
    print("清理Tasks")
    print("=" * 80)
    print()
    
    lock_dir = output_path / ".locks"
    
    for task in tasks:
        print(f"  清理 Task {task.task_id}...")
        
        if dry_run:
            print(f"    [DRY RUN] 将删除: {task.task_dir}")
        else:
            # 删除task目录
            if task.task_dir.exists():
                import shutil
                shutil.rmtree(task.task_dir)
                print(f"    ✓ 已删除: {task.task_dir}")
        
        # 删除锁文件
        lock_file = lock_dir / f"task_{task.task_id}.lock"
        if lock_file.exists():
            if dry_run:
                print(f"    [DRY RUN] 将删除锁: {lock_file}")
            else:
                lock_file.unlink()
                print(f"    ✓ 已删除锁: {lock_file}")
    
    if not dry_run:
        print()
        print(f"✓ 已清理 {len(tasks)} 个tasks")


def main():
    parser = argparse.ArgumentParser(
        description='增量修复不完整的tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查不完整的tasks
  python fix_incomplete_tasks.py --output-path output
  
  # 只修复info.json问题
  python fix_incomplete_tasks.py --output-path output --fix-info-json
  
  # 清理所有不完整的tasks
  python fix_incomplete_tasks.py --output-path output --clean
  
  # 清理特定类别的tasks
  python fix_incomplete_tasks.py --output-path output --clean --category all_missing
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
        help='数据集路径（用于修复）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只检查，不执行操作'
    )
    
    parser.add_argument(
        '--fix-info-json',
        action='store_true',
        help='修复info.json问题'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='清理不完整的tasks'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        choices=['only_info_json_content', 'only_meta_files', 'only_videos', 'only_parquet',
                 'meta_and_videos', 'meta_and_parquet', 'videos_and_parquet', 'all_missing', 'all'],
        default='all',
        help='要处理的类别（默认: all）'
    )
    
    args = parser.parse_args()
    
    if not args.output_path.exists():
        print(f"❌ 输出目录不存在: {args.output_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("检查不完整的Tasks")
    print("=" * 80)
    print()
    
    # 查找不完整的tasks
    print(f"扫描输出目录: {args.output_path}")
    incomplete_tasks = find_incomplete_tasks(args.output_path)
    
    if not incomplete_tasks:
        print("\n✅ 所有tasks都完整！")
        return
    
    # 分类
    categories = categorize_tasks(incomplete_tasks)
    
    # 打印摘要
    print_task_summary(incomplete_tasks, categories)
    
    # 建议修复策略
    suggest_fix_strategy(categories)
    
    # 执行操作
    if args.fix_info_json:
        # 修复info.json
        tasks_to_fix = categories['only_info_json_content']
        if not tasks_to_fix:
            print("没有需要修复info.json的tasks")
            return
        
        print("\n" + "=" * 80)
        print("修复 info.json")
        print("=" * 80)
        print()
        
        success_count = 0
        for task in tasks_to_fix:
            if fix_info_json(task, args.dataset_path, dry_run=args.dry_run):
                success_count += 1
        
        print()
        print(f"✓ 成功修复 {success_count}/{len(tasks_to_fix)} 个tasks")
    
    elif args.clean:
        # 清理tasks
        if args.category == 'all':
            tasks_to_clean = incomplete_tasks
        else:
            tasks_to_clean = categories[args.category]
        
        if not tasks_to_clean:
            print(f"没有需要清理的tasks（类别: {args.category}）")
            return
        
        clean_tasks(tasks_to_clean, args.output_path, dry_run=args.dry_run)
    
    print()


if __name__ == '__main__':
    main()
