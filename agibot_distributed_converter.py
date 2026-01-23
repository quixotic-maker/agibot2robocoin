#!/usr/bin/env python3
"""
Distributed Agibot Dataset Converter

A standalone script to convert Agibot raw data format to LeRobot format.
Supports distributed processing across multiple compute nodes with file-based coordination.

This converter is designed to run on up to 20 compute nodes simultaneously, with automatic
task coordination using file-based locking. It processes robot demonstration data from the
Agibot format (H5 files + videos) and converts it to the LeRobot format (Parquet + videos).

Key Features:
    - Standalone script with minimal dependencies (only lerobot + standard library)
    - Distributed processing with file-based task coordination
    - Episode-level parallel processing within each task
    - Automatic recovery from node failures
    - Memory-efficient lazy video loading
    - Comprehensive error handling and logging
    - Test mode for local validation

Architecture:
    - AgibotDataReader: Reads and parses Agibot raw data format
    - DistributedTaskCoordinator: Manages task distribution across nodes
    - EpisodeConverter: Converts individual episodes
    - LeRobotDatasetWriter: Writes data in LeRobot format
    - TaskProcessor: Coordinates parallel episode processing
    - DistributedConverter: Main orchestrator

Usage:
    Basic usage:
        python agibot_distributed_converter.py \\
            --dataset-path /data/agibot \\
            --output-path /output/lerobot \\
            --repo-id agibot/dataset
    
    Test mode:
        python agibot_distributed_converter.py \\
            --dataset-path /data/agibot \\
            --output-path /output/lerobot \\
            --repo-id agibot/dataset \\
            --test-mode \\
            --max-tasks 2 \\
            --max-episodes 5

Requirements:
    - Python 3.8+
    - lerobot package
    - h5py
    - numpy
    - pandas
    - pyarrow
    - av (PyAV)
    - psutil

Author: Agibot Converter Team
Version: 1.0.0
"""

import argparse
import fcntl
import json
import logging
import os
import psutil
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from functools import wraps

import h5py
import numpy as np


# ============================================================================
# Memory Monitoring Utility
# ============================================================================

class MemoryMonitor:
    """
    Monitor memory usage and log warnings when thresholds are exceeded.
    
    This utility helps track memory consumption during conversion to prevent
    out-of-memory errors and identify memory leaks.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        warning_threshold_mb: float = 8000.0,  # 8 GB
        critical_threshold_mb: float = 12000.0,  # 12 GB
        log_interval_seconds: float = 60.0  # Log every 60 seconds
    ):
        """
        Initialize memory monitor.
        
        Args:
            logger: Logger instance for logging
            warning_threshold_mb: Memory usage threshold for warnings (MB)
            critical_threshold_mb: Memory usage threshold for critical warnings (MB)
            log_interval_seconds: Interval between periodic memory logs (seconds)
        """
        self.logger = logger
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.log_interval_seconds = log_interval_seconds
        
        # Track last log time
        self._last_log_time = 0.0
        
        # Track peak memory usage
        self._peak_memory_mb = 0.0
        
        try:
            self.process = psutil.Process()
            self.logger.info(
                f"Memory monitor initialized: "
                f"warning={warning_threshold_mb}MB, "
                f"critical={critical_threshold_mb}MB"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize memory monitor: {e}")
            self.process = None
    
    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            float: Memory usage in MB (RSS - Resident Set Size)
        """
        if self.process is None:
            return 0.0
        
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def check_memory(self, context: str = "") -> None:
        """
        Check current memory usage and log warnings if thresholds are exceeded.
        
        Args:
            context: Context string to include in log messages (e.g., "Processing task 355")
        """
        if self.process is None:
            return
        
        current_memory_mb = self.get_memory_usage_mb()
        
        # Update peak memory
        if current_memory_mb > self._peak_memory_mb:
            self._peak_memory_mb = current_memory_mb
        
        # Check thresholds
        if current_memory_mb >= self.critical_threshold_mb:
            self.logger.error(
                f"CRITICAL: Memory usage is very high: {current_memory_mb:.1f} MB "
                f"(threshold: {self.critical_threshold_mb} MB) | Context: {context}"
            )
        elif current_memory_mb >= self.warning_threshold_mb:
            self.logger.warning(
                f"WARNING: Memory usage is high: {current_memory_mb:.1f} MB "
                f"(threshold: {self.warning_threshold_mb} MB) | Context: {context}"
            )
    
    def log_memory_periodic(self, context: str = "") -> None:
        """
        Log memory usage periodically based on log_interval_seconds.
        
        Args:
            context: Context string to include in log messages
        """
        if self.process is None:
            return
        
        current_time = time.time()
        
        # Check if enough time has passed since last log
        if current_time - self._last_log_time >= self.log_interval_seconds:
            current_memory_mb = self.get_memory_usage_mb()
            
            self.logger.info(
                f"Memory usage: {current_memory_mb:.1f} MB "
                f"(peak: {self._peak_memory_mb:.1f} MB) | Context: {context}"
            )
            
            self._last_log_time = current_time
            
            # Also check thresholds
            self.check_memory(context)
    
    def get_peak_memory_mb(self) -> float:
        """
        Get peak memory usage since monitor initialization.
        
        Returns:
            float: Peak memory usage in MB
        """
        return self._peak_memory_mb
    
    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        self._peak_memory_mb = self.get_memory_usage_mb()


# ============================================================================
# Retry Utility
# ============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exception types to catch and retry
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        # Get logger if available (from self or args)
                        logger = None
                        if args and hasattr(args[0], 'logger'):
                            logger = args[0].logger
                        
                        if logger:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                                f"Retrying in {delay:.2f}s..."
                            )
                        
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Last attempt failed, re-raise
                        if logger:
                            logger.error(
                                f"All {max_retries} attempts failed for {func.__name__}: {e}"
                            )
                        raise
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# Custom Exception Classes for Error Classification
# ============================================================================

class ConverterError(Exception):
    """Base exception for all converter errors."""
    def __init__(self, message: str, error_type: str = "unknown"):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class ConfigurationError(ConverterError):
    """Exception for configuration-related errors (invalid parameters, missing required parameters)."""
    def __init__(self, message: str):
        super().__init__(message, error_type="configuration")


class DataStructureError(ConverterError):
    """Exception for data structure errors (invalid Agibot dataset structure)."""
    def __init__(self, message: str):
        super().__init__(message, error_type="data_structure")


class FileAccessError(ConverterError):
    """Exception for file access errors (missing or corrupted data files)."""
    def __init__(self, message: str, file_path: Optional[Path] = None):
        super().__init__(message, error_type="file_access")
        self.file_path = file_path


class DataFormatError(ConverterError):
    """Exception for data format errors (invalid H5 structure, missing required fields)."""
    def __init__(self, message: str, expected_format: Optional[str] = None):
        super().__init__(message, error_type="data_format")
        self.expected_format = expected_format


class ConversionError(ConverterError):
    """Exception for conversion errors (failures during data transformation)."""
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_type="conversion")
        self.context = context or {}


class CoordinationError(ConverterError):
    """Exception for coordination errors (lock acquisition failures, status file corruption)."""
    def __init__(self, message: str, retry_possible: bool = True):
        super().__init__(message, error_type="coordination")
        self.retry_possible = retry_possible


# ============================================================================
# AgibotDataReader - Reads and parses Agibot raw data format
# ============================================================================

class AgibotDataReader:
    """
    Reads and parses Agibot raw data format.
    
    Responsible for:
    - Parsing task_info JSON files
    - Locating and validating video files
    - Reading H5 proprio_stats files
    - Extracting state, action, and timestamp data
    - Validating data completeness
    """
    
    def __init__(self, dataset_path: Path, logger: logging.Logger):
        """
        Initialize the Agibot data reader.
        
        Args:
            dataset_path: Path to the Agibot dataset root directory
            logger: Logger instance for logging
        """
        self.dataset_path = dataset_path
        self.logger = logger
        
        # H5 file handle cache for reuse during episode processing
        self._h5_handles = {}  # Dict[Path, h5py.File]
        
        # Validate dataset structure
        self._validate_dataset_structure()
    
    def _validate_dataset_structure(self) -> None:
        """
        Validate that the dataset has the required directory structure.
        
        Raises:
            DataStructureError: If required directories are missing
        """
        required_dirs = ['task_info', 'observations', 'proprio_stats']
        
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                raise DataStructureError(
                    f"Required directory '{dir_name}' not found in dataset path: {self.dataset_path}"
                )
            if not dir_path.is_dir():
                raise DataStructureError(
                    f"'{dir_name}' exists but is not a directory: {dir_path}"
                )
        
        self.logger.info(f"✓ Dataset structure validated: {self.dataset_path}")
    
    def get_all_tasks(self) -> List[int]:
        """
        Scan task_info directory and return list of all task IDs.
        
        Returns:
            List[int]: Sorted list of task IDs found in the dataset
        """
        task_info_dir = self.dataset_path / 'task_info'
        task_ids = []
        
        # Scan for task_*.json files
        for json_file in task_info_dir.glob('task_*.json'):
            try:
                # Extract task ID from filename (e.g., task_355.json -> 355)
                task_id_str = json_file.stem.replace('task_', '')
                task_id = int(task_id_str)
                task_ids.append(task_id)
            except ValueError:
                self.logger.warning(f"Skipping invalid task file: {json_file.name}")
                continue
        
        task_ids.sort()
        self.logger.info(f"Found {len(task_ids)} tasks in dataset")
        
        return task_ids
    
    @retry_with_backoff(max_retries=2, initial_delay=0.5, exceptions=(OSError, IOError))
    def get_task_info(self, task_id: int) -> Dict:
        """
        Parse task info JSON file for a given task ID.
        
        Args:
            task_id: Task ID to read
        
        Returns:
            Dict: Parsed task information containing episodes and metadata
        
        Raises:
            FileAccessError: If task info file doesn't exist
            DataFormatError: If JSON parsing fails
        """
        task_file = self.dataset_path / 'task_info' / f'task_{task_id}.json'
        
        if not task_file.exists():
            raise FileAccessError(
                f"Task info file not found: {task_file}",
                file_path=task_file
            )
        
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            self.logger.debug(f"Loaded task info for task {task_id}: {len(task_data)} episodes")
            return task_data
        
        except json.JSONDecodeError as e:
            raise DataFormatError(
                f"Failed to parse JSON from {task_file}: {e}",
                expected_format="valid JSON array"
            )
    
    def get_episode_ids(self, task_id: int) -> List[int]:
        """
        Extract list of episode IDs from task info.
        
        Args:
            task_id: Task ID to get episodes for
        
        Returns:
            List[int]: List of episode IDs for this task
        """
        task_info = self.get_task_info(task_id)
        
        episode_ids = []
        for episode_data in task_info:
            if 'episode_id' in episode_data:
                episode_ids.append(episode_data['episode_id'])
            else:
                self.logger.warning(f"Episode data missing 'episode_id' field in task {task_id}")
        
        self.logger.debug(f"Task {task_id} has {len(episode_ids)} episodes")
        return episode_ids
    
    def get_video_paths(self, task_id: int, episode_id: int) -> Dict[str, Path]:
        """
        Get paths to all video files for a given episode.
        
        Args:
            task_id: Task ID
            episode_id: Episode ID
        
        Returns:
            Dict[str, Path]: Dictionary mapping camera names to video file paths
        """
        video_dir = self.dataset_path / 'observations' / str(task_id) / str(episode_id) / 'videos'
        
        if not video_dir.exists():
            self.logger.warning(f"Video directory not found: {video_dir}")
            return {}
        
        video_paths = {}
        for video_file in video_dir.glob('*.mp4'):
            # Use stem (filename without extension) as camera name
            camera_name = video_file.stem
            video_paths[camera_name] = video_file
        
        self.logger.debug(f"Found {len(video_paths)} videos for task {task_id}, episode {episode_id}")
        return video_paths
    
    def get_proprio_path(self, task_id: int, episode_id: int) -> Path:
        """
        Get path to proprio_stats H5 file for a given episode.
        
        Args:
            task_id: Task ID
            episode_id: Episode ID
        
        Returns:
            Path: Path to proprio_stats.h5 file
        """
        proprio_path = (
            self.dataset_path / 'proprio_stats' / str(task_id) / str(episode_id) / 'proprio_stats.h5'
        )
        return proprio_path
    
    @retry_with_backoff(max_retries=2, initial_delay=0.5, exceptions=(OSError, IOError))
    def read_proprio_stats(self, task_id: int, episode_id: int) -> Dict[str, np.ndarray]:
        """
        Read proprio_stats H5 file and extract state, action, and timestamp data.
        
        Args:
            task_id: Task ID
            episode_id: Episode ID
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'state_joint_position': Joint state positions
                - 'state_end_position': End effector state positions
                - 'state_head_position': Head state positions
                - 'state_effector_position': Effector state positions (optional, may be empty)
                - 'state_waist_position': Waist state positions (optional, may be empty)
                - 'action_joint_position': Joint action positions
                - 'action_end_position': End effector action positions
                - 'action_head_position': Head action positions
                - 'action_effector_position': Effector action positions (optional, may be empty)
                - 'action_waist_position': Waist action positions (optional, may be empty)
                - 'action_robot_velocity': Robot velocity (optional, may be empty)
                - 'timestamp': Timestamp data
        
        Raises:
            FileAccessError: If H5 file doesn't exist
            DataFormatError: If required fields are missing from H5 file
        """
        proprio_path = self.get_proprio_path(task_id, episode_id)
        
        if not proprio_path.exists():
            raise FileAccessError(
                f"Proprio stats file not found: {proprio_path}",
                file_path=proprio_path
            )
        
        try:
            # Try to reuse existing handle if available
            h5_file = self._get_h5_handle(proprio_path)
            
            # Define required fields (must exist)
            required_fields = {
                'state/joint/position': 'state_joint_position',
                'state/end/position': 'state_end_position',
                'state/head/position': 'state_head_position',
                'action/joint/position': 'action_joint_position',
                'action/end/position': 'action_end_position',
                'action/head/position': 'action_head_position',
                'timestamp': 'timestamp',
            }
            
            # Define optional fields (may not exist in all datasets)
            optional_fields = {
                'state/effector/position': 'state_effector_position',
                'state/waist/position': 'state_waist_position',
                'action/effector/position': 'action_effector_position',
                'action/waist/position': 'action_waist_position',
                'action/robot/velocity': 'action_robot_velocity',
            }
            
            # Check for missing required fields
            missing_required = []
            for h5_path in required_fields.keys():
                if h5_path not in h5_file:
                    missing_required.append(h5_path)
            
            if missing_required:
                raise DataFormatError(
                    f"Missing required fields in {proprio_path}: {', '.join(missing_required)}",
                    expected_format="H5 file with state/action/timestamp fields"
                )
            
            # Extract required data
            data = {}
            for h5_path, key_name in required_fields.items():
                data[key_name] = h5_file[h5_path][:]
            
            # Get number of frames from timestamp
            num_frames = data['timestamp'].shape[0]
            
            # Extract optional data (use empty arrays if not present)
            for h5_path, key_name in optional_fields.items():
                if h5_path in h5_file:
                    data[key_name] = h5_file[h5_path][:]
                    self.logger.debug(f"Found optional field {h5_path}: shape={data[key_name].shape}")
                else:
                    # Create empty array with correct shape (num_frames, 0)
                    data[key_name] = np.empty((num_frames, 0), dtype=np.float32)
                    self.logger.debug(f"Optional field {h5_path} not found, using empty array")
            
            # Log data shapes for debugging
            self.logger.debug(
                f"Loaded proprio stats for task {task_id}, episode {episode_id}: "
                f"{num_frames} frames"
            )
            
            return data
        
        except OSError as e:
            raise FileAccessError(
                f"Failed to read H5 file {proprio_path}: {e}",
                file_path=proprio_path
            )
    
    def _get_h5_handle(self, h5_path: Path) -> h5py.File:
        """
        Get or create an H5 file handle for reuse.
        
        Args:
            h5_path: Path to H5 file
        
        Returns:
            h5py.File: Open H5 file handle
        """
        if h5_path not in self._h5_handles:
            self._h5_handles[h5_path] = h5py.File(h5_path, 'r')
            self.logger.debug(f"Opened H5 file: {h5_path}")
        return self._h5_handles[h5_path]
    
    def close_h5_handles(self) -> None:
        """
        Close all open H5 file handles.
        
        This should be called after episode processing is complete to release resources.
        """
        for h5_path, h5_file in self._h5_handles.items():
            try:
                h5_file.close()
                self.logger.debug(f"Closed H5 file: {h5_path}")
            except Exception as e:
                self.logger.warning(f"Error closing H5 file {h5_path}: {e}")
        
        self._h5_handles.clear()
        self.logger.debug("All H5 file handles closed")
    
    def validate_episode(self, task_id: int, episode_id: int) -> bool:
        """
        Validate that an episode has all required data files.
        
        Args:
            task_id: Task ID
            episode_id: Episode ID
        
        Returns:
            bool: True if episode is valid, False otherwise
        """
        try:
            # Check proprio stats file exists
            proprio_path = self.get_proprio_path(task_id, episode_id)
            if not proprio_path.exists():
                self.logger.warning(f"Episode {episode_id} missing proprio stats file")
                return False
            
            # Check video directory exists and has videos
            video_paths = self.get_video_paths(task_id, episode_id)
            if not video_paths:
                self.logger.warning(f"Episode {episode_id} has no video files")
                return False
            
            return True
        
        except Exception as e:
            self.logger.warning(f"Episode {episode_id} validation failed: {e}")
            return False


# ============================================================================
# DistributedTaskCoordinator - Manages task distribution across nodes
# ============================================================================

class DistributedTaskCoordinator:
    """
    Manages task distribution across multiple nodes using file-based locking.
    
    Responsible for:
    - Creating and managing task lock files
    - Tracking task completion status
    - Detecting and cleaning up stale locks
    - Providing atomic status file updates
    """
    
    def __init__(self, lock_dir: Path, status_file: Path, output_path: Path, logger: logging.Logger):
        """
        Initialize the distributed task coordinator.
        
        Args:
            lock_dir: Directory for lock files
            status_file: Path to status JSON file
            output_path: Base output path for task directories
            logger: Logger instance for logging
        """
        self.lock_dir = lock_dir
        self.status_file = status_file
        self.output_path = output_path
        self.logger = logger
        
        # Create lock directory if it doesn't exist
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize status file if it doesn't exist
        if not self.status_file.exists():
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_file, 'w') as f:
                json.dump({}, f)
        
        self.logger.info(f"Task coordinator initialized: lock_dir={lock_dir}, status_file={status_file}")
    
    def acquire_task_lock(self, task_id: int, node_id: str, timeout: int = 300) -> bool:
        """
        Attempt to acquire a lock for a task using atomic file creation.
        
        Args:
            task_id: Task ID to lock
            node_id: Node ID attempting to acquire the lock
            timeout: Lock timeout in seconds (not used for acquisition, but stored in metadata)
        
        Returns:
            bool: True if lock was acquired, False if task is already locked
        """
        lock_file = self.lock_dir / f"task_{task_id}.lock"
        
        try:
            # Try to create lock file atomically (exclusive creation)
            # This will fail if the file already exists
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            
            # Write lock metadata
            lock_info = {
                "task_id": task_id,
                "node_id": node_id,
                "timestamp": time.time(),
                "pid": os.getpid()
            }
            
            lock_data = json.dumps(lock_info).encode('utf-8')
            os.write(fd, lock_data)
            os.close(fd)
            
            self.logger.info(f"✓ Acquired lock for task {task_id} (node: {node_id})")
            return True
            
        except FileExistsError:
            # Lock already exists
            self.logger.debug(f"Task {task_id} is already locked")
            return False
        
        except Exception as e:
            self.logger.error(f"Error acquiring lock for task {task_id}: {e}")
            return False
    
    def release_task_lock(self, task_id: int) -> None:
        """
        Release a task lock by deleting the lock file.
        
        Args:
            task_id: Task ID to unlock
        """
        lock_file = self.lock_dir / f"task_{task_id}.lock"
        
        try:
            if lock_file.exists():
                lock_file.unlink()
                self.logger.info(f"✓ Released lock for task {task_id}")
            else:
                self.logger.warning(f"Lock file for task {task_id} does not exist")
        
        except Exception as e:
            self.logger.error(f"Error releasing lock for task {task_id}: {e}")
    
    def is_task_locked(self, task_id: int) -> bool:
        """
        Check if a task is currently locked.
        
        Args:
            task_id: Task ID to check
        
        Returns:
            bool: True if task is locked, False otherwise
        """
        lock_file = self.lock_dir / f"task_{task_id}.lock"
        return lock_file.exists()
    
    def is_task_completed(self, task_id: int) -> bool:
        """
        Check if a task has been completed by checking if its output directory exists.
        
        This is simpler and more reliable than checking status files.
        
        Args:
            task_id: Task ID to check
        
        Returns:
            bool: True if task directory exists, False otherwise
        """
        task_output_dir = self.output_path / f"task_{task_id}"
        return task_output_dir.exists()
    
    def mark_task_completed(
        self,
        task_id: int,
        node_id: str,
        episodes_count: int = 0,
        max_retries: int = 3
    ) -> None:
        """
        Mark a task as completed using atomic JSON update with file locking.
        
        Note: The primary completion indicator is the task directory existence.
        This status file is mainly for logging and monitoring purposes.
        
        Args:
            task_id: Task ID to mark as completed
            node_id: Node ID that completed the task
            episodes_count: Number of episodes processed
            max_retries: Maximum number of retry attempts
        """
        task_key = str(task_id)
        
        for attempt in range(max_retries):
            try:
                # Open file with exclusive lock
                with open(self.status_file, 'r+') as f:
                    # Acquire exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    try:
                        # Read current status
                        f.seek(0)
                        status_data = json.load(f)
                        
                        # Update status
                        status_data[task_key] = {
                            "task_id": task_id,
                            "completed": True,
                            "timestamp": time.time(),
                            "node_id": node_id,
                            "episodes_count": episodes_count
                        }
                        
                        # Write updated status
                        f.seek(0)
                        f.truncate()
                        json.dump(status_data, f, indent=2)
                        
                        self.logger.info(
                            f"✓ Marked task {task_id} as completed "
                            f"(node: {node_id}, episodes: {episodes_count})"
                        )
                        return
                    
                    finally:
                        # Release lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Failed to update status for task {task_id} (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"Failed to update status for task {task_id} after {max_retries} attempts: {e}")
                    raise
    
    def cleanup_stale_locks(self, max_age_seconds: int = 3600) -> None:
        """
        Clean up stale lock files older than the specified age.
        
        Args:
            max_age_seconds: Maximum age of lock files in seconds
        """
        current_time = time.time()
        cleaned_count = 0
        
        for lock_file in self.lock_dir.glob("task_*.lock"):
            try:
                # Read lock metadata
                with open(lock_file, 'r') as f:
                    lock_info = json.load(f)
                
                lock_timestamp = lock_info.get('timestamp', 0)
                lock_age = current_time - lock_timestamp
                
                if lock_age > max_age_seconds:
                    # Lock is stale, remove it
                    lock_file.unlink()
                    task_id = lock_info.get('task_id', 'unknown')
                    self.logger.warning(
                        f"Cleaned up stale lock for task {task_id} "
                        f"(age: {lock_age:.0f}s, node: {lock_info.get('node_id', 'unknown')})"
                    )
                    cleaned_count += 1
            
            except Exception as e:
                self.logger.error(f"Error cleaning up lock file {lock_file}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} stale lock(s)")
    
    def get_available_tasks(self, all_tasks: List[int]) -> List[int]:
        """
        Get list of tasks that are available for processing (not completed and not locked).
        
        A task is considered completed if its output directory exists.
        This is simpler and more reliable than checking status files.
        
        Args:
            all_tasks: List of all task IDs in the dataset
        
        Returns:
            List[int]: List of available task IDs
        """
        available_tasks = []
        completed_count = 0
        locked_count = 0
        
        for task_id in all_tasks:
            # Check if task directory exists (definitive completion check)
            task_output_dir = self.output_path / f"task_{task_id}"
            if task_output_dir.exists():
                self.logger.debug(f"Task {task_id} directory exists, skipping")
                completed_count += 1
                continue
            
            # Skip if currently locked
            if self.is_task_locked(task_id):
                self.logger.debug(f"Task {task_id} is locked, skipping")
                locked_count += 1
                continue
            
            available_tasks.append(task_id)
        
        self.logger.info(
            f"Found {len(available_tasks)} available tasks out of {len(all_tasks)} total "
            f"(completed: {completed_count}, locked: {locked_count})"
        )
        return available_tasks


# ============================================================================
# Data Classes
# ============================================================================

from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeData:
    """
    Container for episode data after conversion.
    
    Attributes:
        episode_id: Episode identifier
        task_id: Task identifier
        task_name: Human-readable task name
        states: State array (N, state_dim)
        actions: Action array (N, action_dim)
        timestamps: Timestamp array (N,)
        videos: Dictionary mapping camera names to LazyVideoReader instances
        num_frames: Number of frames in the episode
        state_dim_info: Dictionary with state dimension breakdown (joint_dim, end_dim, head_dim)
        action_dim_info: Dictionary with action dimension breakdown (joint_dim, end_dim, head_dim)
    """
    episode_id: int
    task_id: int
    task_name: str
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    videos: Dict[str, Any]  # Dict[str, LazyVideoReader]
    num_frames: int
    state_dim_info: Dict[str, int] = None
    action_dim_info: Dict[str, int] = None


@dataclass
class TaskResult:
    """
    Container for task processing results.
    
    Attributes:
        task_id: Task identifier
        success: Whether the task completed successfully
        episodes_processed: Number of episodes successfully processed
        episodes_skipped: Number of episodes skipped (already exist)
        episodes_failed: Number of episodes that failed to process
        error_message: Error message if task failed (None if successful)
    """
    task_id: int
    success: bool
    episodes_processed: int
    episodes_skipped: int
    episodes_failed: int
    error_message: Optional[str]


class LazyVideoReader:
    """
    Lazy video reader that loads frames on demand to minimize memory usage.
    
    This wrapper provides frame-by-frame access to video files without loading
    the entire video into memory at once. Includes frame caching for sequential access.
    """
    
    def __init__(self, video_path: Path, logger: logging.Logger, cache_size: int = 10):
        """
        Initialize lazy video reader.
        
        Args:
            video_path: Path to video file
            logger: Logger instance
            cache_size: Number of frames to cache for sequential access (default: 10)
        """
        self.video_path = video_path
        self.logger = logger
        self._container = None
        self._stream = None
        self._frame_count = None
        
        # Frame cache for sequential access
        self._cache_size = cache_size
        self._frame_cache = {}  # Dict[int, np.ndarray]
        self._cache_order = []  # List to track LRU order
        
        # Validate video file exists
        if not video_path.exists():
            raise FileAccessError(
                f"Video file not found: {video_path}",
                file_path=video_path
            )
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self):
        """Open the video file for reading."""
        try:
            import av
            self._container = av.open(str(self.video_path))
            self._stream = self._container.streams.video[0]
            self._frame_count = self._stream.frames
            self.logger.debug(f"Opened video: {self.video_path.name} ({self._frame_count} frames)")
        except Exception as e:
            self.logger.error(f"Failed to open video {self.video_path}: {e}")
            raise
    
    def close(self):
        """Close the video file and clear cache."""
        if self._container:
            self._container.close()
            self._container = None
            self._stream = None
        
        # Clear frame cache
        self._frame_cache.clear()
        self._cache_order.clear()
    
    def get_frame_count(self) -> int:
        """
        Get the number of frames in the video.
        
        Returns:
            int: Number of frames
        """
        if self._frame_count is None:
            # Open temporarily to get frame count
            was_closed = self._container is None
            if was_closed:
                self.open()
            frame_count = self._frame_count
            if was_closed:
                self.close()
            return frame_count
        return self._frame_count
    
    def read_frame(self, frame_idx: int) -> np.ndarray:
        """
        Read a specific frame from the video with caching for sequential access.
        
        Note: This method uses seek which may not be frame-accurate for all videos.
        For sequential access, iterate through decode() instead.
        
        Args:
            frame_idx: Frame index to read
        
        Returns:
            np.ndarray: Frame as numpy array (H, W, C)
        """
        # Check cache first
        if frame_idx in self._frame_cache:
            self.logger.debug(f"Cache hit for frame {frame_idx}")
            # Update LRU order
            self._cache_order.remove(frame_idx)
            self._cache_order.append(frame_idx)
            return self._frame_cache[frame_idx]
        
        if self._container is None:
            raise RuntimeError("Video not opened. Call open() first or use context manager.")
        
        try:
            # Seek to frame
            self._container.seek(frame_idx, stream=self._stream)
            
            # Read frame
            for frame in self._container.decode(video=0):
                frame_array = frame.to_ndarray(format='rgb24')
                
                # Add to cache
                self._add_to_cache(frame_idx, frame_array)
                
                return frame_array
            
            raise ValueError(f"Could not read frame {frame_idx} from {self.video_path}")
        
        except Exception as e:
            self.logger.error(f"Error reading frame {frame_idx} from {self.video_path}: {e}")
            raise
    
    def _add_to_cache(self, frame_idx: int, frame_array: np.ndarray) -> None:
        """
        Add a frame to the cache using LRU eviction policy.
        
        Args:
            frame_idx: Frame index
            frame_array: Frame data
        """
        # If cache is full, remove oldest frame
        if len(self._frame_cache) >= self._cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._frame_cache[oldest_idx]
        
        # Add new frame
        self._frame_cache[frame_idx] = frame_array
        self._cache_order.append(frame_idx)
    
    def __repr__(self):
        return f"LazyVideoReader({self.video_path.name})"


# ============================================================================
# EpisodeConverter - Converts single episodes from Agibot to LeRobot format
# ============================================================================

class EpisodeConverter:
    """
    Converts a single episode from Agibot format to LeRobot format.
    
    Responsible for:
    - Converting single episode data
    - Processing video encoding with lazy loading
    - Processing state and action arrays
    - Synchronizing timestamps across modalities
    - Generating episode metadata
    """
    
    def __init__(self, data_reader: AgibotDataReader, logger: logging.Logger):
        """
        Initialize the episode converter.
        
        Args:
            data_reader: AgibotDataReader instance for reading source data
            logger: Logger instance for logging
        """
        self.data_reader = data_reader
        self.logger = logger
    
    def _process_videos(self, video_paths: Dict[str, Path]) -> Dict[str, LazyVideoReader]:
        """
        Create LazyVideoReader wrappers for all camera videos.
        
        Args:
            video_paths: Dictionary mapping camera names to video file paths
        
        Returns:
            Dict[str, LazyVideoReader]: Dictionary mapping camera names to video readers
        """
        video_readers = {}
        
        for camera_name, video_path in video_paths.items():
            try:
                # Create lazy video reader
                reader = LazyVideoReader(video_path, self.logger)
                video_readers[camera_name] = reader
                self.logger.debug(f"Created video reader for camera: {camera_name}")
            
            except FileAccessError as e:
                # Gracefully handle missing video files (file access error)
                self.logger.warning(f"Video file not found for camera {camera_name}: {video_path}")
                continue
            
            except Exception as e:
                # Log error but continue with other videos (conversion error)
                self.logger.error(f"Error creating video reader for camera {camera_name}: {e}")
                continue
        
        if not video_readers:
            self.logger.warning("No video readers created - all video files missing or failed to load")
        else:
            self.logger.info(f"Created {len(video_readers)} video readers")
        
        return video_readers
    
    def _process_states(self, proprio_data: Dict[str, np.ndarray]) -> tuple[np.ndarray, Dict[str, int]]:
        """
        Concatenate state arrays from proprio data.
        
        Handles optional fields (effector, waist) which may have 0 dimensions.
        
        Args:
            proprio_data: Dictionary containing state data arrays
        
        Returns:
            tuple: (concatenated state array (N, state_dim), dimension info dict)
        """
        # Extract state components
        state_joint = proprio_data['state_joint_position']
        state_end = proprio_data['state_end_position']
        state_head = proprio_data['state_head_position']
        state_effector = proprio_data['state_effector_position']
        state_waist = proprio_data['state_waist_position']
        
        # Flatten 3D arrays to 2D (N, features)
        # state_end might be (N, 2, 3) for dual arms -> flatten to (N, 6)
        if state_end.ndim == 3:
            state_end = state_end.reshape(state_end.shape[0], -1)
        
        # Track dimensions for each component
        dim_info = {
            'joint_dim': state_joint.shape[1],
            'end_dim': state_end.shape[1],
            'head_dim': state_head.shape[1],
            'effector_dim': state_effector.shape[1],
            'waist_dim': state_waist.shape[1]
        }
        
        # Concatenate along feature dimension (only include non-empty arrays)
        state_components = [state_joint, state_end, state_head]
        if state_effector.shape[1] > 0:
            state_components.append(state_effector)
        if state_waist.shape[1] > 0:
            state_components.append(state_waist)
        
        states = np.concatenate(state_components, axis=1)
        
        self.logger.debug(
            f"Processed states: joint={state_joint.shape}, end={state_end.shape}, "
            f"head={state_head.shape}, effector={state_effector.shape}, waist={state_waist.shape}, "
            f"total={states.shape}"
        )
        
        return states, dim_info
    
    def _process_actions(self, proprio_data: Dict[str, np.ndarray]) -> tuple[np.ndarray, Dict[str, int]]:
        """
        Concatenate action arrays from proprio data.
        
        Handles optional fields (effector, waist, robot_velocity) which may have 0 dimensions.
        
        Args:
            proprio_data: Dictionary containing action data arrays
        
        Returns:
            tuple: (concatenated action array (N, action_dim), dimension info dict)
        """
        # Extract action components
        action_joint = proprio_data['action_joint_position']
        action_end = proprio_data['action_end_position']
        action_head = proprio_data['action_head_position']
        action_effector = proprio_data['action_effector_position']
        action_waist = proprio_data['action_waist_position']
        action_robot_velocity = proprio_data['action_robot_velocity']
        
        # Flatten 3D arrays to 2D (N, features)
        # action_end might be (N, 2, 3) for dual arms -> flatten to (N, 6)
        if action_end.ndim == 3:
            action_end = action_end.reshape(action_end.shape[0], -1)
        
        # Track dimensions for each component
        dim_info = {
            'joint_dim': action_joint.shape[1],
            'end_dim': action_end.shape[1],
            'head_dim': action_head.shape[1],
            'effector_dim': action_effector.shape[1],
            'waist_dim': action_waist.shape[1],
            'robot_velocity_dim': action_robot_velocity.shape[1]
        }
        
        # Concatenate along feature dimension (only include non-empty arrays)
        action_components = [action_joint, action_end, action_head]
        if action_effector.shape[1] > 0:
            action_components.append(action_effector)
        if action_waist.shape[1] > 0:
            action_components.append(action_waist)
        if action_robot_velocity.shape[1] > 0:
            action_components.append(action_robot_velocity)
        
        actions = np.concatenate(action_components, axis=1)
        
        self.logger.debug(
            f"Processed actions: joint={action_joint.shape}, end={action_end.shape}, "
            f"head={action_head.shape}, effector={action_effector.shape}, waist={action_waist.shape}, "
            f"robot_velocity={action_robot_velocity.shape}, total={actions.shape}"
        )
        
        return actions, dim_info
    
    def _synchronize_timestamps(
        self,
        timestamps: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        video_readers: Dict[str, LazyVideoReader]
    ) -> Dict[str, Any]:
        """
        Synchronize timestamps and validate frame counts across all modalities.
        
        Args:
            timestamps: Timestamp array
            states: State array
            actions: Action array
            video_readers: Dictionary of video readers
        
        Returns:
            Dict containing synchronized data and validation results
        
        Raises:
            ConversionError: If frame counts don't match across modalities
        """
        # Get frame counts
        num_timestamp_frames = len(timestamps)
        num_state_frames = len(states)
        num_action_frames = len(actions)
        
        # Validate state and action frame counts match timestamps
        if num_state_frames != num_timestamp_frames:
            raise ConversionError(
                f"State frame count ({num_state_frames}) does not match "
                f"timestamp count ({num_timestamp_frames})",
                context={
                    "state_frames": num_state_frames,
                    "timestamp_frames": num_timestamp_frames,
                    "action_frames": num_action_frames
                }
            )
        
        if num_action_frames != num_timestamp_frames:
            raise ConversionError(
                f"Action frame count ({num_action_frames}) does not match "
                f"timestamp count ({num_timestamp_frames})",
                context={
                    "state_frames": num_state_frames,
                    "timestamp_frames": num_timestamp_frames,
                    "action_frames": num_action_frames
                }
            )
        
        # Validate video frame counts (if videos exist)
        video_frame_counts = {}
        for camera_name, reader in video_readers.items():
            try:
                frame_count = reader.get_frame_count()
                video_frame_counts[camera_name] = frame_count
                
                # Check if video frame count matches
                if frame_count != num_timestamp_frames:
                    self.logger.warning(
                        f"Video frame count mismatch for camera {camera_name}: "
                        f"video has {frame_count} frames, expected {num_timestamp_frames}"
                    )
            except Exception as e:
                self.logger.error(f"Error getting frame count for camera {camera_name}: {e}")
        
        self.logger.info(
            f"Timestamp synchronization: {num_timestamp_frames} frames "
            f"(states: {num_state_frames}, actions: {num_action_frames}, "
            f"videos: {len(video_frame_counts)})"
        )
        
        return {
            'num_frames': num_timestamp_frames,
            'timestamps': timestamps,
            'states': states,
            'actions': actions,
            'video_frame_counts': video_frame_counts
        }
    
    def convert_episode(
        self,
        task_id: int,
        episode_id: int,
        task_name: str = ""
    ) -> EpisodeData:
        """
        Convert a single episode from Agibot format to LeRobot format.
        
        This method coordinates all conversion steps:
        1. Read proprio stats (states, actions, timestamps)
        2. Get video paths and create lazy readers
        3. Process states and actions
        4. Synchronize timestamps across modalities
        5. Create EpisodeData container
        
        Args:
            task_id: Task ID
            episode_id: Episode ID
            task_name: Human-readable task name (optional)
        
        Returns:
            EpisodeData: Converted episode data
        
        Raises:
            FileAccessError: If required data files are missing
            DataFormatError: If data format is invalid
            ConversionError: If data conversion fails
        """
        self.logger.info(f"Converting episode {episode_id} from task {task_id}")
        
        try:
            # Step 1: Read proprio stats
            self.logger.debug("Reading proprio stats...")
            proprio_data = self.data_reader.read_proprio_stats(task_id, episode_id)
            
            # Step 2: Get video paths and create readers
            self.logger.debug("Processing videos...")
            video_paths = self.data_reader.get_video_paths(task_id, episode_id)
            video_readers = self._process_videos(video_paths)
            
            # Step 3: Process states and actions
            self.logger.debug("Processing states and actions...")
            states, state_dim_info = self._process_states(proprio_data)
            actions, action_dim_info = self._process_actions(proprio_data)
            
            # Step 4: Synchronize timestamps
            self.logger.debug("Synchronizing timestamps...")
            timestamps = proprio_data['timestamp']
            sync_result = self._synchronize_timestamps(
                timestamps, states, actions, video_readers
            )
            
            # Step 5: Create EpisodeData container
            episode_data = EpisodeData(
                episode_id=episode_id,
                task_id=task_id,
                task_name=task_name,
                states=sync_result['states'],
                actions=sync_result['actions'],
                timestamps=sync_result['timestamps'],
                videos=video_readers,
                num_frames=sync_result['num_frames'],
                state_dim_info=state_dim_info,
                action_dim_info=action_dim_info
            )
            
            self.logger.info(
                f"✓ Successfully converted episode {episode_id}: "
                f"{episode_data.num_frames} frames, {len(video_readers)} videos"
            )
            
            return episode_data
        
        except (FileAccessError, DataFormatError, ConversionError) as e:
            # Re-raise known error types with context
            self.logger.error(
                f"Failed to convert episode {episode_id} from task {task_id}: "
                f"[{e.error_type}] {e}",
                exc_info=True
            )
            raise
        
        except Exception as e:
            # Wrap unknown errors as ConversionError
            self.logger.error(
                f"Unexpected error converting episode {episode_id} from task {task_id}: {e}",
                exc_info=True
            )
            raise ConversionError(
                f"Unexpected error during episode conversion: {e}",
                context={"task_id": task_id, "episode_id": episode_id}
            )


# ============================================================================
# LeRobotDatasetWriter - Writes converted data in LeRobot format
# ============================================================================

class LeRobotDatasetWriter:
    """
    Writes converted data in LeRobot format.
    
    Responsible for:
    - Creating LeRobot directory structure (data/, videos/, images/, meta/)
    - Writing Parquet files for each episode to data/chunk-XXX/
    - Encoding and writing video files to videos/chunk-XXX/{camera_name}/
    - Generating meta/info.json (dataset metadata and features definition)
    - Generating meta/episodes.jsonl (episode index)
    - Generating meta/tasks.jsonl (task index)
    - Generating meta/episodes_stats.jsonl (episode statistics)
    """
    
    def __init__(
        self,
        output_path: Path,
        task_id: int,
        repo_id: str,
        fps: int = 30,
        logger: logging.Logger = None
    ):
        """
        Initialize the LeRobot dataset writer.
        
        Args:
            output_path: Path to root output directory
            task_id: Task ID for this dataset
            repo_id: LeRobot repository identifier
            fps: Frames per second for videos
            logger: Logger instance for logging
        """
        # Create task-specific output directory
        self.output_path = output_path / f"task_{task_id}"
        self.task_id = task_id
        self.repo_id = repo_id
        self.fps = fps
        self.logger = logger or logging.getLogger(__name__)
        
        # Track episodes and tasks for metadata generation
        self.episodes_info = []
        self.tasks_info = {}
        self.episode_count = 0
        self.total_frames = 0
        
        # Track failed episodes for error reporting
        self.failed_episodes = []  # List of {episode_id, episode_index, task_id, error_message}
        
        # Track dimension info from first episode
        self.state_dim_info = None
        self.action_dim_info = None
        
        # Directory paths (under task-specific directory)
        self.data_dir = self.output_path / "data"
        self.videos_dir = self.output_path / "videos"
        self.images_dir = self.output_path / "images"
        self.meta_dir = self.output_path / "meta"
        
        self.logger.info(f"LeRobotDatasetWriter initialized: task_id={task_id}, output_path={self.output_path}, repo_id={repo_id}")
    
    def initialize_dataset(self, features: Dict = None) -> None:
        """
        Initialize the dataset by creating the directory structure.
        
        Creates the following directories:
        - data/: For Parquet files
        - videos/: For encoded video files
        - images/: For image files (may be empty if using videos)
        - meta/: For metadata files
        
        Args:
            features: Optional features definition (will be generated if not provided)
        """
        self.logger.info("Initializing LeRobot dataset structure...")
        
        # Create main directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"✓ Created directory structure at {self.output_path}")
        self.logger.debug(f"  - data/: {self.data_dir}")
        self.logger.debug(f"  - videos/: {self.videos_dir}")
        self.logger.debug(f"  - images/: {self.images_dir}")
        self.logger.debug(f"  - meta/: {self.meta_dir}")
    
    def write_episode(
        self,
        episode_data: EpisodeData,
        episode_index: int,
        chunk_size: int = 1000
    ) -> None:
        """
        Write a single episode to the dataset.
        
        This method:
        1. Writes Parquet file with state, action, timestamp, and index data
        2. Encodes and writes video files for all cameras
        3. Updates episode tracking for metadata generation
        
        Args:
            episode_data: EpisodeData container with all episode information
            episode_index: Global episode index (0-based)
            chunk_size: Number of episodes per chunk directory
        """
        self.logger.info(f"Writing episode {episode_index} (task {episode_data.task_id}, episode {episode_data.episode_id})")
        
        # Determine chunk number
        chunk_num = episode_index // chunk_size
        chunk_name = f"chunk-{chunk_num:03d}"
        
        # Create chunk directories
        data_chunk_dir = self.data_dir / chunk_name
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Write Parquet file
        self._write_parquet(episode_data, episode_index, data_chunk_dir)
        
        # Write video files
        self._write_videos(episode_data, episode_index, chunk_name)
        
        # Update tracking
        self.episodes_info.append({
            "episode_index": episode_index,
            "task_id": episode_data.task_id,
            "episode_id": episode_data.episode_id,
            "task_name": episode_data.task_name,
            "length": episode_data.num_frames
        })
        
        # Track dimension info from first episode
        if self.state_dim_info is None and episode_data.state_dim_info:
            self.state_dim_info = episode_data.state_dim_info
        if self.action_dim_info is None and episode_data.action_dim_info:
            self.action_dim_info = episode_data.action_dim_info
        
        # Track task info
        if episode_data.task_id not in self.tasks_info:
            self.tasks_info[episode_data.task_id] = {
                "task_index": len(self.tasks_info),
                "task_id": episode_data.task_id,
                "task_name": episode_data.task_name
            }
        
        self.episode_count += 1
        self.total_frames += episode_data.num_frames
        
        self.logger.info(f"✓ Episode {episode_index} written successfully")
    
    @retry_with_backoff(max_retries=2, initial_delay=0.5, exceptions=(OSError, IOError))
    def _write_parquet(
        self,
        episode_data: EpisodeData,
        episode_index: int,
        data_chunk_dir: Path,
        use_buffering: bool = True
    ) -> None:
        """
        Write episode data to Parquet file with nested array format.
        
        Format matches LeRobot standard:
        - observation.state: column of arrays (one array per frame)
        - action: column of arrays (one array per frame)
        - timestamp, frame_index, episode_index, index, task_index: scalar columns
        
        Args:
            episode_data: Episode data to write
            episode_index: Global episode index
            data_chunk_dir: Directory for this chunk's data files
            use_buffering: Whether to use buffered writing (default: True)
        """
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                f"Required package not available: {e}. "
                "Please install pandas and pyarrow: pip install pandas pyarrow"
            )
        
        # Prepare data dictionary
        num_frames = episode_data.num_frames
        
        # Create frame indices
        frame_indices = np.arange(num_frames, dtype=np.int64)
        episode_indices = np.full(num_frames, episode_index, dtype=np.int64)
        
        # Create global index (frame_index within episode)
        indices = np.arange(num_frames, dtype=np.int64)
        
        # Task index (always 0 for per-task datasets)
        task_indices = np.zeros(num_frames, dtype=np.int64)
        
        # Build data dictionary with nested arrays
        # observation.state and action are stored as arrays (one per row)
        data_dict = {
            "observation.state": list(episode_data.states.astype(np.float32)),  # List of arrays
            "action": list(episode_data.actions.astype(np.float32)),  # List of arrays
            "timestamp": episode_data.timestamps.astype(np.float32),
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "index": indices,
            "task_index": task_indices
        }
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Write to Parquet with buffering options
        parquet_file = data_chunk_dir / f"episode_{episode_index:06d}.parquet"
        table = pa.Table.from_pandas(df)
        
        if use_buffering:
            # Use buffered writing with compression
            pq.write_table(
                table,
                parquet_file,
                compression='snappy',  # Fast compression
                use_dictionary=True,   # Enable dictionary encoding
                write_statistics=True  # Write column statistics
            )
        else:
            # Simple write without buffering
            pq.write_table(table, parquet_file)
        
        self.logger.debug(
            f"Wrote Parquet file: {parquet_file.name} ({num_frames} frames, "
            f"buffering={'enabled' if use_buffering else 'disabled'})"
        )
    
    def _write_videos(
        self,
        episode_data: EpisodeData,
        episode_index: int,
        chunk_name: str,
        batch_size: int = 30
    ) -> None:
        """
        Encode and write video files for all cameras with batched frame encoding.
        
        Args:
            episode_data: Episode data containing video readers
            episode_index: Global episode index
            chunk_name: Chunk name (e.g., "chunk-000")
            batch_size: Number of frames to batch for encoding (default: 30)
        """
        try:
            import av
        except ImportError:
            raise ImportError(
                "PyAV is required for video encoding. "
                "Please install it: pip install av"
            )
        
        for camera_name, video_reader in episode_data.videos.items():
            # Create camera-specific directory
            camera_dir = self.videos_dir / chunk_name / f"observation.images.{camera_name}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Output video file
            output_video = camera_dir / f"episode_{episode_index:06d}.mp4"
            
            try:
                # Open source video directly for sequential reading
                source_container = av.open(str(video_reader.video_path))
                source_stream = source_container.streams.video[0]
                
                # Create output container
                output_container = av.open(str(output_video), mode='w')
                
                # Create video stream
                stream = output_container.add_stream('h264', rate=self.fps)
                stream.width = source_stream.width
                stream.height = source_stream.height
                stream.pix_fmt = 'yuv420p'
                
                # Read and encode frames sequentially
                frame_count = 0
                frame_batch = []
                
                for packet in source_container.demux(source_stream):
                    for frame in packet.decode():
                        # Convert frame to numpy array and back to VideoFrame
                        # This ensures consistent format
                        frame_array = frame.to_ndarray(format='rgb24')
                        video_frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                        frame_batch.append(video_frame)
                        frame_count += 1
                        
                        # Encode batch when it reaches batch_size
                        if len(frame_batch) >= batch_size:
                            for batch_frame in frame_batch:
                                for pkt in stream.encode(batch_frame):
                                    output_container.mux(pkt)
                            frame_batch.clear()
                
                # Encode remaining frames in batch
                if frame_batch:
                    for batch_frame in frame_batch:
                        for pkt in stream.encode(batch_frame):
                            output_container.mux(pkt)
                    frame_batch.clear()
                
                # Flush encoder
                for pkt in stream.encode():
                    output_container.mux(pkt)
                
                # Close containers
                output_container.close()
                source_container.close()
                
                self.logger.debug(
                    f"Encoded video: {camera_name} -> {output_video.name} "
                    f"({frame_count} frames, batch_size={batch_size})"
                )
            
            except Exception as e:
                self.logger.error(f"Failed to encode video for camera {camera_name}: {e}")
                # Clean up partial file
                if output_video.exists():
                    output_video.unlink()
                raise
    
    def write_meta_info(self, robot_type: str = "agibot") -> None:
        """
        Write meta/info.json with dataset metadata and features definition.
        
        Args:
            robot_type: Type of robot (default: "agibot")
        """
        self.logger.info("Writing meta/info.json...")
        
        # Determine state and action dimensions from first episode
        state_dim = 0
        action_dim = 0
        camera_names = []
        state_names = []
        action_names = []
        
        if self.episodes_info:
            # We need to read back the first parquet file to get dimensions
            first_episode_idx = self.episodes_info[0]["episode_index"]
            chunk_num = first_episode_idx // 1000
            chunk_name = f"chunk-{chunk_num:03d}"
            parquet_file = self.data_dir / chunk_name / f"episode_{first_episode_idx:06d}.parquet"
            
            if parquet_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_parquet(parquet_file)
                    
                    # Get dimensions from nested arrays
                    if 'observation.state' in df.columns:
                        state_dim = len(df['observation.state'].iloc[0])
                    
                    if 'action' in df.columns:
                        action_dim = len(df['action'].iloc[0])
                        
                except Exception as e:
                    self.logger.warning(f"Could not read parquet file for dimensions: {e}")
            
            # Get camera names from videos directory
            chunk_dir = self.videos_dir / chunk_name
            if chunk_dir.exists():
                for camera_dir in chunk_dir.iterdir():
                    if camera_dir.is_dir() and camera_dir.name.startswith("observation.images."):
                        camera_name = camera_dir.name.replace("observation.images.", "")
                        camera_names.append(camera_name)
        
        # Generate meaningful feature names based on dimension info
        if self.state_dim_info:
            joint_dim = self.state_dim_info['joint_dim']
            end_dim = self.state_dim_info['end_dim']
            head_dim = self.state_dim_info['head_dim']
            effector_dim = self.state_dim_info.get('effector_dim', 0)
            waist_dim = self.state_dim_info.get('waist_dim', 0)
            
            # Joint state names (1-indexed)
            for i in range(joint_dim):
                state_names.append(f"joint_{i+1}")
            
            # End effector state names (1-indexed)
            for i in range(end_dim):
                state_names.append(f"end_effector_{i+1}")
            
            # Head state names (1-indexed)
            for i in range(head_dim):
                state_names.append(f"head_{i+1}")
            
            # Effector state names (1-indexed)
            for i in range(effector_dim):
                state_names.append(f"effector_{i+1}")
            
            # Waist state names (1-indexed)
            for i in range(waist_dim):
                state_names.append(f"waist_{i+1}")
        else:
            # Fallback to generic names if dimension info not available (1-indexed)
            state_names = [f"state_{i+1}" for i in range(state_dim)]
        
        if self.action_dim_info:
            joint_dim = self.action_dim_info['joint_dim']
            end_dim = self.action_dim_info['end_dim']
            head_dim = self.action_dim_info['head_dim']
            effector_dim = self.action_dim_info.get('effector_dim', 0)
            waist_dim = self.action_dim_info.get('waist_dim', 0)
            robot_velocity_dim = self.action_dim_info.get('robot_velocity_dim', 0)
            
            # Joint action names (1-indexed)
            for i in range(joint_dim):
                action_names.append(f"joint_{i+1}")
            
            # End effector action names (1-indexed)
            for i in range(end_dim):
                action_names.append(f"end_effector_{i+1}")
            
            # Head action names (1-indexed)
            for i in range(head_dim):
                action_names.append(f"head_{i+1}")
            
            # Effector action names (1-indexed)
            for i in range(effector_dim):
                action_names.append(f"effector_{i+1}")
            
            # Waist action names (1-indexed)
            for i in range(waist_dim):
                action_names.append(f"waist_{i+1}")
            
            # Robot velocity action names (1-indexed)
            for i in range(robot_velocity_dim):
                action_names.append(f"robot_velocity_{i+1}")
        else:
            # Fallback to generic names if dimension info not available (1-indexed)
            action_names = [f"action_{i+1}" for i in range(action_dim)]
        
        # Build features definition
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": state_names
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": action_names
            }
        }
        
        # Add camera features
        for camera_name in camera_names:
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": [480, 640, 3],  # Default shape, actual may vary
                "names": None
            }
        
        # Build info dictionary
        info = {
            "codebase_version": "1.0.0",
            "robot_type": robot_type,
            "total_episodes": self.episode_count,
            "total_frames": self.total_frames,
            "total_tasks": 1,  # Each task dataset contains only one task
            "total_videos": self.episode_count * len(camera_names),
            "total_chunks": (self.episode_count + 999) // 1000,  # Ceiling division
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {
                "train": f"0:{self.episode_count}"
            },
            "data_path": "data/{episode_chunk}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{episode_chunk}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features
        }
        
        # Write to file
        info_file = self.meta_dir / "info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"✓ Wrote meta/info.json ({self.episode_count} episodes, {self.total_frames} frames)")
        self.logger.info(f"  State features: {len(state_names)} ({', '.join(state_names[:5])}...)")
        self.logger.info(f"  Action features: {len(action_names)} ({', '.join(action_names[:5])}...)")
    
    def write_meta_episodes(self) -> None:
        """
        Write meta/episodes.jsonl with episode index information.
        
        Each line contains: {"episode_index": int, "tasks": [str], "length": int}
        """
        self.logger.info("Writing meta/episodes.jsonl...")
        
        episodes_file = self.meta_dir / "episodes.jsonl"
        
        with open(episodes_file, 'w') as f:
            for episode_info in self.episodes_info:
                episode_entry = {
                    "episode_index": episode_info["episode_index"],
                    "tasks": [str(episode_info["task_id"])],
                    "length": episode_info["length"]
                }
                f.write(json.dumps(episode_entry) + '\n')
        
        self.logger.info(f"✓ Wrote meta/episodes.jsonl ({len(self.episodes_info)} episodes)")
    
    def write_meta_tasks(self) -> None:
        """
        Write meta/tasks.jsonl with task index information.
        
        Each line contains: {"task_index": int, "task": str}
        For per-task datasets, this file contains only one entry (the current task).
        """
        self.logger.info("Writing meta/tasks.jsonl...")
        
        tasks_file = self.meta_dir / "tasks.jsonl"
        
        with open(tasks_file, 'w') as f:
            # Write only the current task with task_index = 0
            task_entry = {
                "task_index": 0,  # Always 0 for per-task datasets
                "task": str(self.task_id)
            }
            f.write(json.dumps(task_entry) + '\n')
        
        self.logger.info(f"✓ Wrote meta/tasks.jsonl (task {self.task_id})")
    
    def finalize_dataset(self) -> None:
        """
        Finalize the dataset by generating remaining metadata files.
        
        This method:
        1. Generates meta/episodes_stats.jsonl with episode statistics
        2. Generates episode_source_mapping.json
        3. Generates original_data_paths.json
        """
        self.logger.info("Finalizing dataset...")
        
        # Generate episodes_stats.jsonl
        self._write_episodes_stats()
        
        # Generate source mapping files
        self._write_source_mapping()
        
        self.logger.info("✓ Dataset finalized successfully")
    
    def _write_episodes_stats(self) -> None:
        """Write meta/episodes_stats.jsonl with episode statistics."""
        stats_file = self.meta_dir / "episodes_stats.jsonl"
        
        with open(stats_file, 'w') as f:
            for episode_info in self.episodes_info:
                stats_entry = {
                    "episode_index": episode_info["episode_index"],
                    "num_frames": episode_info["length"],
                    "task_id": episode_info["task_id"],
                    "episode_id": episode_info["episode_id"]
                }
                f.write(json.dumps(stats_entry) + '\n')
        
        self.logger.debug(f"Wrote meta/episodes_stats.jsonl")
    
    def _write_source_mapping(self) -> None:
        """Write episode_source_mapping.json and original_data_paths.json."""
        # Episode source mapping
        source_mapping = {}
        for episode_info in self.episodes_info:
            source_mapping[episode_info["episode_index"]] = {
                "task_id": episode_info["task_id"],
                "episode_id": episode_info["episode_id"],
                "task_name": episode_info["task_name"],
                "status": "success"
            }
        
        # Add failed episodes
        for failed_info in self.failed_episodes:
            source_mapping[f"failed_{failed_info['episode_id']}"] = {
                "task_id": failed_info["task_id"],
                "episode_id": failed_info["episode_id"],
                "status": "failed",
                "error_message": failed_info["error_message"]
            }
        
        mapping_file = self.output_path / "episode_source_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(source_mapping, f, indent=2)
        
        self.logger.debug("Wrote episode_source_mapping.json")
        
        # Original data paths (placeholder for now)
        paths_file = self.output_path / "original_data_paths.json"
        with open(paths_file, 'w') as f:
            json.dump({
                "dataset_path": str(self.output_path),
                "repo_id": self.repo_id
            }, f, indent=2)
        
        self.logger.debug("Wrote original_data_paths.json")


# ============================================================================
# TaskProcessor - Coordinates task-level conversion with parallel episode processing
# ============================================================================

class TaskProcessor:
    """
    Coordinates conversion of a single task with parallel episode processing.
    
    Responsible for:
    - Coordinating episode-level parallel processing
    - Skipping already converted episodes
    - Handling episode conversion errors
    - Aggregating task-level results
    """
    
    def __init__(
        self,
        data_reader: AgibotDataReader,
        max_workers: int,
        logger: logging.Logger
    ):
        """
        Initialize the task processor.
        
        Args:
            data_reader: AgibotDataReader instance for reading source data
            max_workers: Maximum number of parallel workers for episode processing
            logger: Logger instance for logging
        """
        self.data_reader = data_reader
        self.max_workers = max_workers
        self.logger = logger
        
        # Create episode converter
        self.episode_converter = EpisodeConverter(data_reader, logger)
        
        # Episode locks for concurrent processing
        self._episode_locks = {}
        self._lock_mutex = __import__('threading').Lock()
    
    def _check_episode_exists(self, writer: LeRobotDatasetWriter, episode_index: int, chunk_size: int = 1000) -> bool:
        """
        Check if an episode output already exists and is complete.
        
        This method verifies that both the Parquet file and video files exist
        for the given episode.
        
        Args:
            writer: LeRobotDatasetWriter instance to check
            episode_index: Episode index to check (0-based within task)
            chunk_size: Number of episodes per chunk directory
        
        Returns:
            bool: True if episode output exists and is complete, False otherwise
        """
        # Determine chunk number
        chunk_num = episode_index // chunk_size
        chunk_name = f"chunk-{chunk_num:03d}"
        
        # Check Parquet file
        data_chunk_dir = writer.data_dir / chunk_name
        parquet_file = data_chunk_dir / f"episode_{episode_index:06d}.parquet"
        
        if not parquet_file.exists():
            self.logger.debug(f"Episode {episode_index}: Parquet file not found")
            return False
        
        # Check video files (at least one camera should exist)
        videos_chunk_dir = writer.videos_dir / chunk_name
        if not videos_chunk_dir.exists():
            self.logger.debug(f"Episode {episode_index}: Videos directory not found")
            return False
        
        # Look for any video files for this episode
        video_found = False
        for camera_dir in videos_chunk_dir.iterdir():
            if camera_dir.is_dir():
                video_file = camera_dir / f"episode_{episode_index:06d}.mp4"
                if video_file.exists():
                    video_found = True
                    break
        
        if not video_found:
            self.logger.debug(f"Episode {episode_index}: No video files found")
            return False
        
        self.logger.debug(f"Episode {episode_index}: Output exists and is complete")
        return True
    
    def _cleanup_partial_episode(
        self,
        writer: LeRobotDatasetWriter,
        episode_index: int,
        episode_id: int,
        chunk_size: int = 1000
    ) -> None:
        """
        Clean up partial data for a failed episode.
        
        This method removes:
        - Parquet file
        - All video files for this episode
        
        Args:
            writer: LeRobotDatasetWriter instance
            episode_index: Episode index (0-based within task)
            episode_id: Original episode ID from source data
            chunk_size: Number of episodes per chunk directory
        """
        self.logger.info(f"Cleaning up partial data for episode {episode_id} (index {episode_index})")
        
        # Determine chunk number
        chunk_num = episode_index // chunk_size
        chunk_name = f"chunk-{chunk_num:03d}"
        
        files_removed = 0
        
        # Remove Parquet file
        data_chunk_dir = writer.data_dir / chunk_name
        parquet_file = data_chunk_dir / f"episode_{episode_index:06d}.parquet"
        
        if parquet_file.exists():
            try:
                parquet_file.unlink()
                files_removed += 1
                self.logger.debug(f"Removed Parquet file: {parquet_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove Parquet file {parquet_file}: {e}")
        
        # Remove video files
        videos_chunk_dir = writer.videos_dir / chunk_name
        if videos_chunk_dir.exists():
            for camera_dir in videos_chunk_dir.iterdir():
                if camera_dir.is_dir():
                    video_file = camera_dir / f"episode_{episode_index:06d}.mp4"
                    if video_file.exists():
                        try:
                            video_file.unlink()
                            files_removed += 1
                            self.logger.debug(f"Removed video file: {video_file}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove video file {video_file}: {e}")
        
        if files_removed > 0:
            self.logger.info(f"✓ Cleaned up {files_removed} file(s) for episode {episode_id}")
        else:
            self.logger.debug(f"No partial files found for episode {episode_id}")
    
    def _process_episode_parallel_with_writer(
        self,
        task_id: int,
        episode_ids: List[int],
        task_name: str,
        writer: LeRobotDatasetWriter,
        max_episodes: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Process episodes in parallel using ThreadPoolExecutor with a specific writer.
        
        Args:
            task_id: Task ID being processed
            episode_ids: List of episode IDs to process
            task_name: Human-readable task name
            writer: LeRobotDatasetWriter instance for this task
            max_episodes: Maximum number of episodes to process (None for all)
        
        Returns:
            Dict with counts: processed, skipped, failed
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Limit episodes if max_episodes is specified
        if max_episodes is not None:
            episode_ids = episode_ids[:max_episodes]
        
        results = {
            'processed': 0,
            'skipped': 0,
            'failed': 0
        }
        
        def process_single_episode(episode_id: int, episode_index: int) -> tuple:
            """
            Process a single episode.
            
            Returns:
                tuple: (status, episode_id, error_message)
                status: 'processed', 'skipped', or 'failed'
            """
            try:
                # Acquire episode lock to prevent concurrent processing
                with self._lock_mutex:
                    if episode_id in self._episode_locks:
                        self.logger.debug(f"Episode {episode_id} is locked by another worker")
                        return ('skipped', episode_id, None)
                    self._episode_locks[episode_id] = True
                
                try:
                    # Check if episode already exists
                    if self._check_episode_exists(writer, episode_index):
                        self.logger.info(f"Episode {episode_id} already exists, skipping")
                        return ('skipped', episode_id, None)
                    
                    # Convert episode
                    self.logger.info(f"Processing episode {episode_id} (index {episode_index})")
                    episode_data = self.episode_converter.convert_episode(
                        task_id, episode_id, task_name
                    )
                    
                    # Write episode
                    writer.write_episode(episode_data, episode_index)
                    
                    self.logger.info(f"✓ Episode {episode_id} processed successfully")
                    return ('processed', episode_id, None)
                
                finally:
                    # Release episode lock
                    with self._lock_mutex:
                        if episode_id in self._episode_locks:
                            del self._episode_locks[episode_id]
            
            except Exception as e:
                error_msg = f"Failed to process episode {episode_id}: {e}"
                
                # Log error with context
                error_context = {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "error_type": getattr(e, 'error_type', 'unknown'),
                    "error_message": str(e)
                }
                
                # Add file path if it's a FileAccessError
                if isinstance(e, FileAccessError) and hasattr(e, 'file_path'):
                    error_context["file_path"] = str(e.file_path)
                
                # Add data shapes if it's a ConversionError
                if isinstance(e, ConversionError) and hasattr(e, 'context'):
                    error_context.update(e.context)
                
                self.logger.error(
                    f"{error_msg} | Context: {error_context}",
                    exc_info=True
                )
                
                # Clean up partial data for this episode
                try:
                    self._cleanup_partial_episode(writer, episode_index, episode_id)
                except Exception as cleanup_error:
                    self.logger.error(f"Failed to cleanup partial data for episode {episode_id}: {cleanup_error}")
                
                return ('failed', episode_id, error_msg)
        
        # Process episodes in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all episodes (episode_index starts from 0)
            futures = {}
            for i, episode_id in enumerate(episode_ids):
                episode_index = i  # 0-based index within this task
                future = executor.submit(process_single_episode, episode_id, episode_index)
                futures[future] = episode_id
            
            # Collect results as they complete
            for future in as_completed(futures):
                episode_id = futures[future]
                try:
                    status, _, error_msg = future.result()
                    results[status] += 1
                    
                    if status == 'failed' and error_msg:
                        self.logger.error(f"Episode {episode_id} failed: {error_msg}")
                        
                        # Record failed episode in writer
                        writer.failed_episodes.append({
                            "episode_id": episode_id,
                            "task_id": task_id,
                            "error_message": error_msg
                        })
                
                except Exception as e:
                    self.logger.error(f"Unexpected error processing episode {episode_id}: {e}")
                    results['failed'] += 1
        
        return results
    
    def process_task(
        self,
        task_id: int,
        output_path: Path,
        repo_id: str,
        fps: int = 30,
        max_episodes: Optional[int] = None
    ) -> TaskResult:
        """
        Process a complete task by converting all its episodes.
        
        This method:
        1. Creates a task-specific LeRobotDatasetWriter
        2. Initializes the task output directory
        3. Gets all episode IDs for the task
        4. Processes episodes in parallel (episode_index starts from 0)
        5. Writes task metadata
        6. Handles errors gracefully (continues with other episodes)
        7. Returns aggregated results
        8. Closes H5 file handles after processing
        
        Args:
            task_id: Task ID to process
            output_path: Root output path (task subdirectory will be created)
            repo_id: LeRobot repository identifier
            fps: Frames per second for videos
            max_episodes: Maximum number of episodes to process (None for all)
        
        Returns:
            TaskResult: Aggregated results for the task
        """
        self.logger.info(f"Processing task {task_id}")
        
        try:
            # Create task-specific writer
            task_writer = LeRobotDatasetWriter(
                output_path=output_path,
                task_id=task_id,
                repo_id=repo_id,
                fps=fps,
                logger=self.logger
            )
            
            # Initialize task dataset
            task_writer.initialize_dataset()
            
            # Get task info
            task_info = self.data_reader.get_task_info(task_id)
            task_name = ""
            if task_info and len(task_info) > 0:
                task_name = task_info[0].get('task_name', f'Task {task_id}')
            
            # Get episode IDs
            episode_ids = self.data_reader.get_episode_ids(task_id)
            
            if not episode_ids:
                self.logger.warning(f"Task {task_id} has no episodes")
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    episodes_processed=0,
                    episodes_skipped=0,
                    episodes_failed=0,
                    error_message=None
                )
            
            total_episodes = len(episode_ids)
            limited_episodes = min(total_episodes, max_episodes) if max_episodes else total_episodes
            
            self.logger.info(
                f"Task {task_id} ({task_name}): {total_episodes} episodes found, "
                f"processing {limited_episodes} episodes"
            )
            
            # Process episodes in parallel (episode_index starts from 0 for each task)
            results = self._process_episode_parallel_with_writer(
                task_id,
                episode_ids,
                task_name,
                task_writer,
                max_episodes
            )
            
            # Write task metadata
            if results['processed'] > 0:
                self.logger.info(f"Writing metadata for task {task_id}...")
                task_writer.write_meta_info()
                task_writer.write_meta_episodes()
                task_writer.write_meta_tasks()
                task_writer.finalize_dataset()
            
            # Determine success
            success = results['failed'] == 0
            error_message = None
            if not success:
                error_message = f"{results['failed']} episodes failed to process"
            
            # Create task result
            task_result = TaskResult(
                task_id=task_id,
                success=success,
                episodes_processed=results['processed'],
                episodes_skipped=results['skipped'],
                episodes_failed=results['failed'],
                error_message=error_message
            )
            
            # Log detailed progress
            self.logger.info(
                f"✓ Task {task_id} completed: "
                f"processed={results['processed']}, "
                f"skipped={results['skipped']}, "
                f"failed={results['failed']} | "
                f"Task: {task_name}"
            )
            
            return task_result
        
        except Exception as e:
            error_msg = f"Failed to process task {task_id}: {e}"
            
            # Log error with context
            error_context = {
                "task_id": task_id,
                "max_episodes": max_episodes,
                "error_type": getattr(e, 'error_type', 'unknown')
            }
            
            self.logger.error(f"{error_msg} | Context: {error_context}", exc_info=True)
            
            return TaskResult(
                task_id=task_id,
                success=False,
                episodes_processed=0,
                episodes_skipped=0,
                episodes_failed=0,
                error_message=error_msg
            )
        
        finally:
            # Always close H5 file handles after task processing
            self.data_reader.close_h5_handles()


def detect_node_id() -> str:
    """
    Detect node ID from environment variable or hostname.
    
    This function attempts to identify the current compute node using multiple strategies:
    1. First, check for NODE_ID environment variable (set by orchestration systems)
    2. Fall back to hostname (unique per node in most clusters)
    3. Last resort: use process ID (for local testing)
    
    Returns:
        str: Node identifier string
        
    Examples:
        >>> os.environ['NODE_ID'] = 'compute-node-01'
        >>> detect_node_id()
        'compute-node-01'
        
        >>> del os.environ['NODE_ID']
        >>> detect_node_id()  # Returns hostname
        'my-hostname'
    """
    # Try environment variable first
    node_id = os.environ.get('NODE_ID')
    if node_id:
        return node_id
    
    # Fall back to hostname
    try:
        hostname = socket.gethostname()
        return hostname
    except Exception:
        # Last resort: use process ID
        return f"node_{os.getpid()}"


def setup_logging(
    log_dir: Path,
    node_id: str,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Set up logging with file and console handlers, including separate error and progress logs.
    
    This function creates a comprehensive logging setup with:
    - Console handler: For real-time monitoring
    - Main log file: Complete log with all levels (DEBUG and above)
    - Error log file: Only ERROR and CRITICAL messages
    - Progress log file: Filtered INFO messages with progress indicators
    
    Log files are named with node ID and timestamp for easy identification in distributed environments.
    
    Args:
        log_dir: Directory for log files (will be created if it doesn't exist)
        node_id: Node identifier for log file naming (e.g., "node-01")
        log_level: Logging level for console output (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logging.Logger: Configured logger instance
        
    Examples:
        >>> logger = setup_logging(Path("logs"), "node-01", "INFO")
        >>> logger.info("Processing started")
        [2024-01-22 10:30:00] [INFO] [agibot_converter] Processing started
        
    Note:
        If the logger already has handlers (e.g., from a previous call), this function
        returns the existing logger without adding duplicate handlers.
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('agibot_converter')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simple formatter for progress logs
    progress_formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler - main log
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_log_file = log_dir / f"node_{node_id}_{timestamp}.log"
    file_handler = logging.FileHandler(main_log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # File handler - error log (only ERROR and above)
    error_log_file = log_dir / f"node_{node_id}_{timestamp}_errors.log"
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # File handler - progress log (INFO and above, for tracking progress)
    progress_log_file = log_dir / f"node_{node_id}_{timestamp}_progress.log"
    progress_handler = logging.FileHandler(progress_log_file)
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(progress_formatter)
    
    # Create a filter to only log progress-related messages
    class ProgressFilter(logging.Filter):
        def filter(self, record):
            # Log messages that contain progress indicators
            progress_keywords = ['✓', 'Processing', 'Completed', 'Starting', 'Finalizing', 'converted', 'processed']
            return any(keyword in record.getMessage() for keyword in progress_keywords)
    
    progress_handler.addFilter(ProgressFilter())
    logger.addHandler(progress_handler)
    
    logger.info(f"Logging initialized for node {node_id}")
    logger.info(f"Main log: {main_log_file}")
    logger.info(f"Error log: {error_log_file}")
    logger.info(f"Progress log: {progress_log_file}")
    
    return logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the distributed converter.
    
    This function defines and parses all command-line arguments needed to run the converter.
    It provides comprehensive help text and examples for users.
    
    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - dataset_path (Path): Input Agibot dataset directory
            - output_path (Path): Output LeRobot dataset directory
            - repo_id (str): LeRobot repository identifier
            - max_workers (int): Number of parallel workers (default: 4)
            - test_mode (bool): Enable test mode (default: False)
            - max_tasks (int|None): Maximum tasks to process (default: None)
            - max_episodes (int|None): Maximum episodes per task (default: None)
            - log_level (str): Logging level (default: INFO)
            - lock_timeout (int): Lock timeout in seconds (default: 3600)
    
    Examples:
        >>> args = parse_arguments()  # From command line
        >>> print(args.dataset_path)
        /data/agibot
        
    Raises:
        SystemExit: If required arguments are missing or --help is requested
    """
    parser = argparse.ArgumentParser(
        description='Distributed Agibot Dataset Converter - Convert Agibot raw data to LeRobot format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python agibot_distributed_converter.py \\
    --dataset-path /data/agibot \\
    --output-path /output/lerobot \\
    --repo-id agibot/dataset

  # Test mode with limited tasks
  python agibot_distributed_converter.py \\
    --dataset-path /data/agibot \\
    --output-path /output/lerobot \\
    --repo-id agibot/dataset \\
    --test-mode \\
    --max-tasks 2 \\
    --max-episodes 5

  # Custom parallelism and logging
  python agibot_distributed_converter.py \\
    --dataset-path /data/agibot \\
    --output-path /output/lerobot \\
    --repo-id agibot/dataset \\
    --max-workers 4 \\
    --log-level DEBUG
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset-path',
        type=Path,
        required=True,
        help='Path to input Agibot dataset directory'
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        required=True,
        help='Path to output LeRobot dataset directory'
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='LeRobot repository identifier (e.g., agibot/dataset)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of parallel workers for episode processing (default: 4)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Enable test mode for single-node testing with limited processing'
    )
    
    parser.add_argument(
        '--max-tasks',
        type=int,
        default=None,
        help='Maximum number of tasks to process (for testing)'
    )
    
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=None,
        help='Maximum number of episodes per task (for testing)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--lock-timeout',
        type=int,
        default=3600,
        help='Lock file timeout in seconds (default: 3600)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Validate command line arguments before starting conversion.
    
    This function performs comprehensive validation of all arguments to ensure:
    - Dataset path exists and is a directory
    - Output path can be created and is writable
    - Numeric parameters are within valid ranges
    - Configuration is consistent
    
    Args:
        args: Parsed arguments from parse_arguments()
        logger: Logger instance for logging validation results
    
    Raises:
        ConfigurationError: If any validation check fails, with a descriptive error message
        
    Examples:
        >>> args = parse_arguments()
        >>> logger = logging.getLogger()
        >>> validate_arguments(args, logger)  # Raises ConfigurationError if invalid
        
    Validation Checks:
        - dataset_path must exist and be a directory
        - output_path must be creatable and writable
        - max_workers must be >= 1
        - max_tasks (if provided) must be >= 1
        - max_episodes (if provided) must be >= 1
        - lock_timeout must be >= 1
    """
    logger.info("Validating command line arguments...")
    
    # Validate dataset path exists
    if not args.dataset_path.exists():
        raise ConfigurationError(f"Dataset path does not exist: {args.dataset_path}")
    
    if not args.dataset_path.is_dir():
        raise ConfigurationError(f"Dataset path is not a directory: {args.dataset_path}")
    
    # Validate output path is writable
    try:
        args.output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ConfigurationError(f"Cannot create output directory {args.output_path}: {e}")
    
    # Validate max_workers
    if args.max_workers < 1:
        raise ConfigurationError(f"max_workers must be >= 1, got {args.max_workers}")
    
    # Validate max_tasks if provided
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ConfigurationError(f"max_tasks must be >= 1, got {args.max_tasks}")
    
    # Validate max_episodes if provided
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ConfigurationError(f"max_episodes must be >= 1, got {args.max_episodes}")
    
    # Validate lock_timeout
    if args.lock_timeout < 1:
        raise ConfigurationError(f"lock_timeout must be >= 1, got {args.lock_timeout}")
    
    logger.info("✓ All arguments validated successfully")


# ============================================================================
# DistributedConverter - Main class coordinating distributed conversion
# ============================================================================

class DistributedConverter:
    """
    Main class coordinating the distributed conversion process.
    
    Responsible for:
    - Parsing command line arguments
    - Initializing all components
    - Main conversion loop
    - Error handling and logging
    - Resource cleanup
    """
    
    def __init__(
        self,
        dataset_path: Path,
        output_path: Path,
        repo_id: str,
        max_workers: int = 4,
        test_mode: bool = False,
        max_tasks: Optional[int] = None,
        max_episodes: Optional[int] = None,
        lock_timeout: int = 3600,
        log_level: str = "INFO",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the distributed converter.
        
        Args:
            dataset_path: Path to input Agibot dataset
            output_path: Path to output LeRobot dataset
            repo_id: LeRobot repository identifier
            max_workers: Number of parallel workers for episode processing
            test_mode: Enable test mode for limited processing
            max_tasks: Maximum number of tasks to process (None for all)
            max_episodes: Maximum number of episodes per task (None for all)
            lock_timeout: Lock file timeout in seconds
            log_level: Logging level
            logger: Logger instance (will be created if not provided)
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.repo_id = repo_id
        self.max_workers = max_workers
        self.test_mode = test_mode
        self.max_tasks = max_tasks
        self.max_episodes = max_episodes
        self.lock_timeout = lock_timeout
        self.log_level = log_level
        
        # Detect node ID
        self.node_id = detect_node_id()
        
        # Set up logging if not provided
        if logger is None:
            log_dir = output_path / "logs"
            self.logger = setup_logging(log_dir, self.node_id, log_level)
        else:
            self.logger = logger
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(
            self.logger,
            warning_threshold_mb=8000.0,  # 8 GB
            critical_threshold_mb=12000.0,  # 12 GB
            log_interval_seconds=60.0  # Log every 60 seconds
        )
        
        # Components (initialized in _initialize_components)
        self.data_reader: Optional[AgibotDataReader] = None
        self.coordinator: Optional[DistributedTaskCoordinator] = None
        self.processor: Optional[TaskProcessor] = None
        
        # Track acquired locks for cleanup
        self._acquired_locks: List[int] = []
        
        self.logger.info(f"DistributedConverter initialized for node {self.node_id}")
    
    def _validate_dataset(self) -> None:
        """
        Validate dataset structure before starting conversion.
        
        This creates the AgibotDataReader which validates the directory structure.
        
        Raises:
            DataStructureError: If dataset structure is invalid
        """
        self.logger.info("Validating dataset structure...")
        
        try:
            # Create data reader (this validates the structure)
            self.data_reader = AgibotDataReader(self.dataset_path, self.logger)
            
            # Get all tasks to ensure dataset is not empty
            all_tasks = self.data_reader.get_all_tasks()
            
            if not all_tasks:
                raise DataStructureError(f"No tasks found in dataset: {self.dataset_path}")
            
            self.logger.info(f"✓ Dataset validated: {len(all_tasks)} tasks found")
        
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            raise
    
    def _initialize_components(self) -> None:
        """
        Initialize all components (reader, coordinator, writer, processor).
        
        This method sets up:
        - AgibotDataReader: For reading source data
        - DistributedTaskCoordinator: For task coordination
        - LeRobotDatasetWriter: For writing output
        - TaskProcessor: For processing tasks
        """
        self.logger.info("Initializing components...")
        
        # Data reader (already created in _validate_dataset)
        if self.data_reader is None:
            self.data_reader = AgibotDataReader(self.dataset_path, self.logger)
        
        # Task coordinator
        lock_dir = self.output_path / ".locks"
        status_file = self.output_path / ".status.json"
        self.coordinator = DistributedTaskCoordinator(
            lock_dir, status_file, self.output_path, self.logger
        )
        
        # Task processor (no longer needs writer)
        self.processor = TaskProcessor(
            self.data_reader,
            self.max_workers,
            self.logger
        )
        
        self.logger.info("✓ All components initialized successfully")
    
    def _cleanup_resources(self) -> None:
        """
        Clean up resources on exit.
        
        This method:
        - Releases all acquired locks
        - Closes file handles
        - Cleans up temporary files
        """
        self.logger.info("Cleaning up resources...")
        
        # Release all acquired locks
        if self.coordinator and self._acquired_locks:
            for task_id in self._acquired_locks:
                try:
                    self.coordinator.release_task_lock(task_id)
                    self.logger.debug(f"Released lock for task {task_id}")
                except Exception as e:
                    self.logger.error(f"Error releasing lock for task {task_id}: {e}")
            
            self._acquired_locks.clear()
        
        self.logger.info("✓ Resource cleanup completed")
    
    def run(self) -> None:
        """
        Run the distributed conversion process.
        
        This is the main entry point that:
        1. Validates the dataset
        2. Initializes all components
        3. Runs the main conversion loop
        4. Finalizes the dataset
        5. Cleans up resources
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("Distributed Agibot Converter Starting")
            self.logger.info("=" * 80)
            self.logger.info(f"Node ID: {self.node_id}")
            self.logger.info(f"Dataset path: {self.dataset_path}")
            self.logger.info(f"Output path: {self.output_path}")
            self.logger.info(f"Repo ID: {self.repo_id}")
            self.logger.info(f"Max workers: {self.max_workers}")
            self.logger.info(f"Test mode: {self.test_mode}")
            if self.max_tasks:
                self.logger.info(f"Max tasks: {self.max_tasks}")
            if self.max_episodes:
                self.logger.info(f"Max episodes: {self.max_episodes}")
            
            # Step 1: Validate dataset
            self._validate_dataset()
            
            # Step 2: Initialize components
            self._initialize_components()
            
            # Step 3: Run main conversion loop
            self._run_conversion_loop()
            
            self.logger.info("=" * 80)
            self.logger.info("Distributed Agibot Converter Completed Successfully")
            self.logger.info("=" * 80)
        
        except Exception as e:
            self.logger.error(f"Fatal error during conversion: {e}", exc_info=True)
            raise
        
        finally:
            # Always clean up resources
            self._cleanup_resources()
    
    def _run_conversion_loop(self) -> None:
        """
        Main conversion loop.
        
        This method:
        - Continuously discovers available tasks
        - Attempts to acquire locks for tasks
        - Processes tasks if lock is acquired
        - Releases locks and updates status on completion
        - Exits when all tasks are complete
        - Monitors memory usage periodically
        """
        self.logger.info("Running conversion loop...")
        
        # Clean up stale locks before starting
        self.coordinator.cleanup_stale_locks(self.lock_timeout)
        
        # Get all tasks from dataset
        all_tasks = self.data_reader.get_all_tasks()
        total_tasks = len(all_tasks)
        
        # Apply max_tasks limit if in test mode
        if self.max_tasks is not None:
            all_tasks = all_tasks[:self.max_tasks]
            self.logger.info(f"Test mode: limiting to {len(all_tasks)} tasks out of {total_tasks}")
        
        # Track statistics
        tasks_processed = 0
        tasks_skipped = 0
        tasks_failed = 0
        total_episodes_processed = 0
        total_episodes_skipped = 0
        total_episodes_failed = 0
        
        # Main loop: process tasks until all are complete
        loop_iteration = 0
        while True:
            loop_iteration += 1
            
            # Periodic memory monitoring
            self.memory_monitor.log_memory_periodic(
                f"Loop iteration {loop_iteration}, tasks processed: {tasks_processed}/{len(all_tasks)}"
            )
            
            # Get available tasks (not completed and not locked)
            available_tasks = self.coordinator.get_available_tasks(all_tasks)
            
            # Log progress
            completed_tasks = total_tasks - len(available_tasks)
            self.logger.info(
                f"Loop iteration {loop_iteration}: "
                f"{completed_tasks}/{total_tasks} tasks completed, "
                f"{len(available_tasks)} tasks available"
            )
            
            if not available_tasks:
                self.logger.info("No more available tasks to process")
                break
            
            # Try to acquire lock for first available task
            task_acquired = False
            current_task_id = None
            
            for task_id in available_tasks:
                if self.coordinator.acquire_task_lock(task_id, self.node_id):
                    # Check if task directory already exists (created by another node)
                    # This is the definitive check - if directory exists, task is done
                    task_output_dir = self.output_path / f"task_{task_id}"
                    if task_output_dir.exists():
                        self.logger.info(
                            f"Task {task_id} directory already exists (processed by another node), skipping"
                        )
                        self.coordinator.release_task_lock(task_id)
                        # Mark as completed in status file
                        self.coordinator.mark_task_completed(task_id, "already_exists", 0)
                        continue
                    
                    task_acquired = True
                    current_task_id = task_id
                    self._acquired_locks.append(task_id)
                    break
            
            if not task_acquired:
                # All available tasks are locked by other nodes
                self.logger.info("All available tasks are locked by other nodes, waiting...")
                
                # Wait a bit before checking again
                time.sleep(5)
                
                # Clean up stale locks
                self.coordinator.cleanup_stale_locks(self.lock_timeout)
                continue
            
            # Process the acquired task
            try:
                self.logger.info(
                    f"Processing task {current_task_id} "
                    f"(progress: {tasks_processed + tasks_failed}/{len(all_tasks)} tasks)"
                )
                
                # Check memory before processing task
                self.memory_monitor.check_memory(f"Before processing task {current_task_id}")
                
                # Process task (each task creates its own writer and starts episode_index from 0)
                task_result = self.processor.process_task(
                    current_task_id,
                    self.output_path,
                    self.repo_id,
                    fps=30,
                    max_episodes=self.max_episodes
                )
                
                # Check memory after processing task
                self.memory_monitor.check_memory(f"After processing task {current_task_id}")
                
                # Update statistics
                if task_result.success:
                    tasks_processed += 1
                    total_episodes_processed += task_result.episodes_processed
                    total_episodes_skipped += task_result.episodes_skipped
                    total_episodes_failed += task_result.episodes_failed
                    
                    self.logger.info(
                        f"✓ Task {current_task_id} completed successfully: "
                        f"processed={task_result.episodes_processed}, "
                        f"skipped={task_result.episodes_skipped}, "
                        f"failed={task_result.episodes_failed} | "
                        f"Overall progress: {tasks_processed}/{len(all_tasks)} tasks, "
                        f"{total_episodes_processed} episodes"
                    )
                else:
                    tasks_failed += 1
                    total_episodes_failed += task_result.episodes_failed
                    
                    self.logger.error(
                        f"✗ Task {current_task_id} failed: {task_result.error_message} | "
                        f"Overall progress: {tasks_processed}/{len(all_tasks)} tasks processed, "
                        f"{tasks_failed} tasks failed"
                    )
                
                # Mark task as completed
                self.coordinator.mark_task_completed(
                    current_task_id,
                    self.node_id,
                    task_result.episodes_processed + task_result.episodes_skipped
                )
            
            except Exception as e:
                tasks_failed += 1
                
                # Log error with context
                error_context = {
                    "task_id": current_task_id,
                    "tasks_processed": tasks_processed,
                    "tasks_failed": tasks_failed
                }
                
                self.logger.error(
                    f"Unexpected error processing task {current_task_id}: {e} | "
                    f"Context: {error_context}",
                    exc_info=True
                )
            
            finally:
                # Always release the lock
                try:
                    self.coordinator.release_task_lock(current_task_id)
                    self._acquired_locks.remove(current_task_id)
                except Exception as e:
                    self.logger.error(
                        f"Error releasing lock for task {current_task_id}: {e}"
                    )
        
        # Log final statistics including memory
        peak_memory_mb = self.memory_monitor.get_peak_memory_mb()
        
        self.logger.info("=" * 80)
        self.logger.info("Conversion Loop Completed")
        self.logger.info(f"Tasks processed: {tasks_processed}")
        self.logger.info(f"Tasks skipped: {tasks_skipped}")
        self.logger.info(f"Tasks failed: {tasks_failed}")
        self.logger.info(f"Total episodes processed: {total_episodes_processed}")
        self.logger.info(f"Total episodes skipped: {total_episodes_skipped}")
        self.logger.info(f"Total episodes failed: {total_episodes_failed}")
        self.logger.info(f"Peak memory usage: {peak_memory_mb:.1f} MB")
        self.logger.info("=" * 80)


def main():
    """
    Main entry point for the distributed converter.
    
    This function:
    1. Parses command-line arguments
    2. Creates a DistributedConverter instance
    3. Runs the conversion process
    4. Handles fatal errors and exits with appropriate status code
    
    Exit Codes:
        0: Success - conversion completed without errors
        1: Failure - fatal error occurred during conversion
        
    Examples:
        Run from command line:
            $ python agibot_distributed_converter.py --dataset-path /data/agibot \\
                --output-path /output/lerobot --repo-id agibot/dataset
        
        Run programmatically:
            >>> if __name__ == "__main__":
            ...     main()
    
    Note:
        This function catches all exceptions and prints them to stderr before exiting.
        Detailed error information is also logged to the error log file.
    """
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Create and run converter
        converter = DistributedConverter(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            repo_id=args.repo_id,
            max_workers=args.max_workers,
            test_mode=args.test_mode,
            max_tasks=args.max_tasks,
            max_episodes=args.max_episodes,
            lock_timeout=args.lock_timeout,
            log_level=args.log_level
        )
        
        converter.run()
        
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
