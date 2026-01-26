#!/usr/bin/env python3
"""
Fix Episode Index Gaps in Converted LeRobot Dataset

This script fixes episode index gaps in already-converted LeRobot datasets.
It renumbers episodes sequentially (0, 1, 2, ...) and updates all metadata files.

Usage:
    python fix_episode_indices.py --dataset-path /path/to/converted/dataset
    python fix_episode_indices.py --dataset-path /path/to/converted/dataset --task-id 355
    python fix_episode_indices.py --dataset-path /path/to/converted/dataset --dry-run

Features:
    - Detects episode index gaps
    - Renumbers episodes sequentially
    - Updates all metadata files (info.json, episodes.jsonl, stats.jsonl)
    - Renames parquet and video files
    - Supports dry-run mode for preview
    - Creates backup before making changes
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def detect_episode_gaps(task_dir: Path, logger: logging.Logger) -> Tuple[List[int], List[int]]:
    """
    Detect episode index gaps by scanning parquet files.
    
    Args:
        task_dir: Path to task directory
        logger: Logger instance
    
    Returns:
        Tuple of (existing_indices, expected_indices)
    """
    logger.info(f"Scanning task directory: {task_dir}")
    
    # Find all episode parquet files
    existing_indices = []
    
    for chunk_dir in (task_dir / "data").glob("chunk-*"):
        for parquet_file in chunk_dir.glob("episode_*.parquet"):
            # Extract episode index from filename
            filename = parquet_file.stem  # e.g., "episode_000002"
            index_str = filename.replace("episode_", "")
            episode_index = int(index_str)
            existing_indices.append(episode_index)
    
    existing_indices.sort()
    
    # Expected indices should be 0, 1, 2, ..., N-1
    expected_indices = list(range(len(existing_indices)))
    
    logger.info(f"Found {len(existing_indices)} episodes")
    logger.info(f"Existing indices: {existing_indices}")
    logger.info(f"Expected indices: {expected_indices}")
    
    return existing_indices, expected_indices


def has_gaps(existing_indices: List[int], expected_indices: List[int]) -> bool:
    """Check if there are gaps in episode indices."""
    return existing_indices != expected_indices


def create_backup(task_dir: Path, logger: logging.Logger) -> Path:
    """
    Create a backup of the task directory.
    
    Args:
        task_dir: Path to task directory
        logger: Logger instance
    
    Returns:
        Path to backup directory
    """
    backup_dir = task_dir.parent / f"{task_dir.name}_backup"
    
    if backup_dir.exists():
        logger.warning(f"Backup already exists: {backup_dir}")
        response = input("Overwrite existing backup? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Backup cancelled")
            sys.exit(0)
        shutil.rmtree(backup_dir)
    
    logger.info(f"Creating backup: {backup_dir}")
    shutil.copytree(task_dir, backup_dir)
    logger.info(f"✓ Backup created: {backup_dir}")
    
    return backup_dir


def rename_episode_files(
    task_dir: Path,
    old_index: int,
    new_index: int,
    logger: logging.Logger,
    dry_run: bool = False
) -> None:
    """
    Rename all files for an episode (parquet and videos).
    
    Args:
        task_dir: Path to task directory
        old_index: Current episode index
        new_index: New episode index
        logger: Logger instance
        dry_run: If True, only print what would be done
    """
    old_name = f"episode_{old_index:06d}"
    new_name = f"episode_{new_index:06d}"
    
    # Rename parquet files
    for chunk_dir in (task_dir / "data").glob("chunk-*"):
        old_parquet = chunk_dir / f"{old_name}.parquet"
        new_parquet = chunk_dir / f"{new_name}.parquet"
        
        if old_parquet.exists():
            if dry_run:
                logger.info(f"  [DRY RUN] Would rename: {old_parquet.name} -> {new_parquet.name}")
            else:
                old_parquet.rename(new_parquet)
                logger.debug(f"  Renamed: {old_parquet.name} -> {new_parquet.name}")
    
    # Rename video files
    videos_dir = task_dir / "videos"
    if videos_dir.exists():
        for chunk_dir in videos_dir.glob("chunk-*"):
            for camera_dir in chunk_dir.iterdir():
                if camera_dir.is_dir():
                    old_video = camera_dir / f"{old_name}.mp4"
                    new_video = camera_dir / f"{new_name}.mp4"
                    
                    if old_video.exists():
                        if dry_run:
                            logger.info(f"  [DRY RUN] Would rename: {old_video.relative_to(task_dir)}")
                        else:
                            old_video.rename(new_video)
                            logger.debug(f"  Renamed: {old_video.relative_to(task_dir)}")


def update_metadata_files(
    task_dir: Path,
    index_mapping: Dict[int, int],
    logger: logging.Logger,
    dry_run: bool = False
) -> None:
    """
    Update all metadata files with new episode indices.
    
    Args:
        task_dir: Path to task directory
        index_mapping: Mapping from old index to new index
        logger: Logger instance
        dry_run: If True, only print what would be done
    """
    meta_dir = task_dir / "meta"
    
    # Update episodes.jsonl
    episodes_file = meta_dir / "episodes.jsonl"
    if episodes_file.exists():
        logger.info("Updating meta/episodes.jsonl...")
        
        updated_lines = []
        with open(episodes_file, 'r') as f:
            for line in f:
                episode_data = json.loads(line)
                old_index = episode_data['episode_index']
                
                if old_index in index_mapping:
                    episode_data['episode_index'] = index_mapping[old_index]
                
                updated_lines.append(json.dumps(episode_data))
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would update {len(updated_lines)} entries")
        else:
            with open(episodes_file, 'w') as f:
                for line in updated_lines:
                    f.write(line + '\n')
            logger.info(f"  ✓ Updated {len(updated_lines)} entries")
    
    # Update stats.jsonl (if exists)
    stats_file = meta_dir / "stats.jsonl"
    if stats_file.exists():
        logger.info("Updating meta/stats.jsonl...")
        
        updated_lines = []
        with open(stats_file, 'r') as f:
            for line in f:
                stats_data = json.loads(line)
                old_index = stats_data['episode_index']
                
                if old_index in index_mapping:
                    stats_data['episode_index'] = index_mapping[old_index]
                
                updated_lines.append(json.dumps(stats_data))
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would update {len(updated_lines)} entries")
        else:
            with open(stats_file, 'w') as f:
                for line in updated_lines:
                    f.write(line + '\n')
            logger.info(f"  ✓ Updated {len(updated_lines)} entries")
    
    # Update info.json (update total_episodes if needed)
    info_file = meta_dir / "info.json"
    if info_file.exists():
        logger.info("Updating meta/info.json...")
        
        with open(info_file, 'r') as f:
            info_data = json.load(f)
        
        # Update total_episodes to match actual count
        actual_count = len(index_mapping)
        if 'total_episodes' in info_data:
            old_count = info_data['total_episodes']
            info_data['total_episodes'] = actual_count
            
            if dry_run:
                logger.info(f"  [DRY RUN] Would update total_episodes: {old_count} -> {actual_count}")
            else:
                with open(info_file, 'w') as f:
                    json.dump(info_data, f, indent=2)
                logger.info(f"  ✓ Updated total_episodes: {old_count} -> {actual_count}")


def fix_task_episodes(
    task_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False,
    create_backup_flag: bool = True
) -> bool:
    """
    Fix episode index gaps for a single task.
    
    Args:
        task_dir: Path to task directory
        logger: Logger instance
        dry_run: If True, only print what would be done
        create_backup_flag: If True, create backup before making changes
    
    Returns:
        bool: True if gaps were fixed, False if no gaps found
    """
    logger.info("=" * 80)
    logger.info(f"Processing task: {task_dir.name}")
    logger.info("=" * 80)
    
    # Detect gaps
    existing_indices, expected_indices = detect_episode_gaps(task_dir, logger)
    
    if not has_gaps(existing_indices, expected_indices):
        logger.info("✓ No gaps found - episodes are already sequential")
        return False
    
    logger.warning("⚠ Gaps detected in episode indices!")
    logger.warning(f"  Existing: {existing_indices}")
    logger.warning(f"  Expected: {expected_indices}")
    
    # Create index mapping
    index_mapping = {old: new for old, new in zip(existing_indices, expected_indices)}
    
    logger.info("\nIndex mapping:")
    for old, new in index_mapping.items():
        if old != new:
            logger.info(f"  Episode {old} -> {new}")
    
    if dry_run:
        logger.info("\n[DRY RUN MODE] - No changes will be made")
        return True
    
    # Confirm with user
    response = input("\nProceed with fixing episode indices? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Operation cancelled")
        return False
    
    # Create backup
    if create_backup_flag:
        create_backup(task_dir, logger)
    
    # Rename files (in reverse order to avoid conflicts)
    logger.info("\nRenaming episode files...")
    for old_index in reversed(existing_indices):
        new_index = index_mapping[old_index]
        if old_index != new_index:
            logger.info(f"Renaming episode {old_index} -> {new_index}")
            rename_episode_files(task_dir, old_index, new_index, logger, dry_run=False)
    
    # Update metadata files
    logger.info("\nUpdating metadata files...")
    update_metadata_files(task_dir, index_mapping, logger, dry_run=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Episode indices fixed successfully!")
    logger.info("=" * 80)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix episode index gaps in converted LeRobot dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to converted dataset root directory"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        help="Specific task ID to fix (if not provided, fixes all tasks)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup (not recommended)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 80)
    logger.info("Episode Index Gap Fixer")
    logger.info("=" * 80)
    logger.info(f"Dataset path: {args.dataset_path}")
    if args.task_id:
        logger.info(f"Task ID: {args.task_id}")
    if args.dry_run:
        logger.info("Mode: DRY RUN (no changes will be made)")
    if args.no_backup:
        logger.warning("Backup disabled - changes will be permanent!")
    
    # Validate dataset path
    if not args.dataset_path.exists():
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)
    
    # Find task directories
    if args.task_id:
        task_dirs = [args.dataset_path / f"task_{args.task_id}"]
    else:
        task_dirs = sorted(args.dataset_path.glob("task_*"))
    
    if not task_dirs:
        logger.error("No task directories found")
        sys.exit(1)
    
    logger.info(f"\nFound {len(task_dirs)} task(s) to process\n")
    
    # Process each task
    fixed_count = 0
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        
        try:
            was_fixed = fix_task_episodes(
                task_dir,
                logger,
                dry_run=args.dry_run,
                create_backup_flag=not args.no_backup
            )
            
            if was_fixed:
                fixed_count += 1
        
        except Exception as e:
            logger.error(f"Error processing {task_dir.name}: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Tasks processed: {len(task_dirs)}")
    logger.info(f"Tasks fixed: {fixed_count}")
    logger.info(f"Tasks already correct: {len(task_dirs) - fixed_count}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN MODE] - No changes were made")
        logger.info("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
