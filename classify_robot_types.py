#!/usr/bin/env python3
"""
根据机器人类型分类LeRobot数据集

区分双手机器人和夹爪机器人，将它们分别组织到不同的文件夹中
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def detect_robot_type(task_dir: Path) -> Tuple[str, str]:
    """
    检测机器人类型
    
    通过检查 meta/info.json 中的 features 来判断是双手还是夹爪
    主要通过 effector 的维度数量来区分：
    - 双手机器人：effector 维度 = 12 (左手6维 + 右手6维)
    - 夹爪机器人：effector 维度 < 12 (通常是2维)
    
    Returns:
        (robot_type, reason)
        robot_type: 'dual_hand', 'gripper', 'unknown'
    """
    meta_info = task_dir / "meta" / "info.json"
    
    if not meta_info.exists():
        return 'unknown', "缺少 meta/info.json"
    
    try:
        with open(meta_info, 'r') as f:
            info = json.load(f)
        
        features = info.get('features', {})
        
        # 方法1: 检查 observation.state 中的 effector 维度
        # 特征名格式: observation.state (包含所有state维度)
        state_feature = features.get('observation.state', {})
        if state_feature:
            state_shape = state_feature.get('shape', [])
            if state_shape:
                state_dim = state_shape[0] if isinstance(state_shape, list) else state_shape
                
                # 根据state总维度判断
                # 双手: joints(14) + end_effector(6) + head(2) + effector(12) + waist(2) + velocity(2) = 38
                # 夹爪: joints(14) + end_effector(6) + head(2) + effector(2) + waist(2) + velocity(2) = 28
                if state_dim >= 36:  # 接近38，考虑可能有些字段缺失
                    return 'dual_hand', f"state维度={state_dim} (>=36, 双手机器人)"
                elif state_dim <= 30:  # 接近28，考虑可能有些字段缺失
                    return 'gripper', f"state维度={state_dim} (<=30, 夹爪机器人)"
        
        # 方法2: 检查 action 维度
        action_feature = features.get('action', {})
        if action_feature:
            action_shape = action_feature.get('shape', [])
            if action_shape:
                action_dim = action_shape[0] if isinstance(action_shape, list) else action_shape
                
                # 双手: joints(14) + end_effector(6) + head(2) + effector(12) + waist(2) + velocity(2) = 38
                # 夹爪: joints(14) + end_effector(6) + head(2) + effector(2) + waist(2) + velocity(2) = 28
                if action_dim >= 36:
                    return 'dual_hand', f"action维度={action_dim} (>=36, 双手机器人)"
                elif action_dim <= 30:
                    return 'gripper', f"action维度={action_dim} (<=30, 夹爪机器人)"
        
        # 方法3: 检查特征名中是否有 left/right 区分
        has_left_right = any('left' in key.lower() or 'right' in key.lower() for key in features.keys())
        if has_left_right:
            return 'dual_hand', "检测到 left/right 特征命名"
        
        return 'unknown', f"无法确定机器人类型 (state_dim={state_shape[0] if state_shape else 'N/A'}, action_dim={action_shape[0] if action_shape else 'N/A'})"
    
    except Exception as e:
        return 'unknown', f"读取失败: {e}"


def classify_tasks(input_dir: Path) -> Dict[str, List[Tuple[int, Path, str]]]:
    """
    分类所有tasks
    
    Returns:
        Dict with keys: 'dual_hand', 'gripper', 'unknown'
        Values: List of (task_id, task_dir, reason)
    """
    classification = {
        'dual_hand': [],
        'gripper': [],
        'unknown': []
    }
    
    # 查找所有task目录
    task_dirs = sorted(input_dir.glob("task_*"))
    
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        
        # 提取task_id
        try:
            task_id = int(task_dir.name.replace("task_", ""))
        except ValueError:
            continue
        
        # 检测机器人类型
        robot_type, reason = detect_robot_type(task_dir)
        
        classification[robot_type].append((task_id, task_dir, reason))
    
    return classification


def organize_by_type(
    input_dir: Path,
    output_base_dir: Path,
    classification: Dict[str, List[Tuple[int, Path, str]]],
    mode: str = 'copy',
    dry_run: bool = False
):
    """
    按机器人类型组织数据集
    
    Args:
        input_dir: 输入目录
        output_base_dir: 输出基础目录
        classification: 分类结果
        mode: 'copy' 或 'move'
        dry_run: 是否只模拟不执行
    """
    # 创建输出目录
    dual_hand_dir = output_base_dir / "dual_hand"
    gripper_dir = output_base_dir / "gripper"
    unknown_dir = output_base_dir / "unknown"
    
    type_dirs = {
        'dual_hand': dual_hand_dir,
        'gripper': gripper_dir,
        'unknown': unknown_dir
    }
    
    # 统计
    stats = {
        'dual_hand': 0,
        'gripper': 0,
        'unknown': 0
    }
    
    for robot_type, tasks in classification.items():
        if not tasks:
            continue
        
        target_dir = type_dirs[robot_type]
        
        print(f"\n处理 {robot_type} 类型 ({len(tasks)} 个tasks):")
        print("=" * 80)
        
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        for task_id, task_dir, reason in tasks:
            target_task_dir = target_dir / task_dir.name
            
            print(f"  Task {task_id}: {reason}")
            
            if dry_run:
                print(f"    [DRY RUN] 将{mode}: {task_dir} -> {target_task_dir}")
            else:
                try:
                    if mode == 'copy':
                        if target_task_dir.exists():
                            print(f"    ⚠️  目标已存在，跳过: {target_task_dir}")
                        else:
                            shutil.copytree(task_dir, target_task_dir)
                            print(f"    ✓ 已复制到: {target_task_dir}")
                            stats[robot_type] += 1
                    elif mode == 'move':
                        if target_task_dir.exists():
                            print(f"    ⚠️  目标已存在，跳过: {target_task_dir}")
                        else:
                            shutil.move(str(task_dir), str(target_task_dir))
                            print(f"    ✓ 已移动到: {target_task_dir}")
                            stats[robot_type] += 1
                except Exception as e:
                    print(f"    ❌ 失败: {e}")
    
    return stats


def copy_metadata_files(input_dir: Path, output_base_dir: Path, dry_run: bool = False):
    """
    复制元数据文件（.locks, logs等）到各个输出目录
    """
    metadata_items = ['.locks', 'logs']
    
    type_dirs = ['dual_hand', 'gripper', 'unknown']
    
    for item_name in metadata_items:
        item_path = input_dir / item_name
        if not item_path.exists():
            continue
        
        for type_dir in type_dirs:
            target_dir = output_base_dir / type_dir
            target_item = target_dir / item_name
            
            if dry_run:
                print(f"[DRY RUN] 将复制: {item_path} -> {target_item}")
            else:
                try:
                    if item_path.is_dir():
                        if not target_item.exists():
                            shutil.copytree(item_path, target_item)
                            print(f"✓ 已复制目录: {item_name} -> {type_dir}/")
                    else:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item_path, target_item)
                        print(f"✓ 已复制文件: {item_name} -> {type_dir}/")
                except Exception as e:
                    print(f"⚠️  复制 {item_name} 到 {type_dir} 失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='根据机器人类型分类LeRobot数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查分类（不执行操作）
  python classify_robot_types.py \\
    --input-dir output \\
    --output-dir output_classified \\
    --dry-run
  
  # 复制到分类目录
  python classify_robot_types.py \\
    --input-dir output \\
    --output-dir output_classified \\
    --mode copy
  
  # 移动到分类目录
  python classify_robot_types.py \\
    --input-dir output \\
    --output-dir output_classified \\
    --mode move
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='输入目录（包含所有task_xxx的目录）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='输出基础目录（将创建 dual_hand/ 和 gripper/ 子目录）'
    )
    
    parser.add_argument(
        '--mode',
        choices=['copy', 'move'],
        default='copy',
        help='操作模式：copy（复制）或 move（移动）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只检查分类，不执行实际操作'
    )
    
    parser.add_argument(
        '--copy-metadata',
        action='store_true',
        help='复制元数据文件（.locks, logs等）'
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"❌ 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("LeRobot数据集机器人类型分类")
    print("=" * 80)
    print()
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"操作模式: {args.mode}")
    print(f"Dry Run: {args.dry_run}")
    print()
    
    # 分类tasks
    print("扫描并分类tasks...")
    classification = classify_tasks(args.input_dir)
    
    # 显示分类结果
    print()
    print("=" * 80)
    print("分类结果")
    print("=" * 80)
    print()
    
    for robot_type in ['dual_hand', 'gripper', 'unknown']:
        tasks = classification[robot_type]
        print(f"{robot_type.upper()}: {len(tasks)} 个tasks")
        
        if tasks:
            for task_id, task_dir, reason in tasks[:5]:  # 只显示前5个
                print(f"  Task {task_id}: {reason}")
            if len(tasks) > 5:
                print(f"  ... 还有 {len(tasks) - 5} 个tasks")
        print()
    
    # 如果只是dry-run，到此结束
    if args.dry_run:
        print("=" * 80)
        print("Dry Run 模式 - 未执行任何操作")
        print("=" * 80)
        print()
        print("要执行分类，请运行:")
        print(f"  python classify_robot_types.py \\")
        print(f"    --input-dir {args.input_dir} \\")
        print(f"    --output-dir {args.output_dir} \\")
        print(f"    --mode {args.mode}")
        return
    
    # 执行组织
    print("=" * 80)
    print("开始组织数据集")
    print("=" * 80)
    
    stats = organize_by_type(
        args.input_dir,
        args.output_dir,
        classification,
        args.mode,
        args.dry_run
    )
    
    # 复制元数据文件
    if args.copy_metadata:
        print()
        print("=" * 80)
        print("复制元数据文件")
        print("=" * 80)
        copy_metadata_files(args.input_dir, args.output_dir, args.dry_run)
    
    # 显示统计
    print()
    print("=" * 80)
    print("完成")
    print("=" * 80)
    print()
    print("统计:")
    print(f"  双手机器人: {stats['dual_hand']} 个tasks -> {args.output_dir}/dual_hand/")
    print(f"  夹爪机器人: {stats['gripper']} 个tasks -> {args.output_dir}/gripper/")
    print(f"  未知类型: {stats['unknown']} 个tasks -> {args.output_dir}/unknown/")
    print()
    
    if stats['unknown'] > 0:
        print("⚠️  有未知类型的tasks，请手动检查 unknown/ 目录")
    
    print("✅ 分类完成!")


if __name__ == '__main__':
    main()
