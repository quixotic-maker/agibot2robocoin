#!/bin/bash

# ========================================
# 多任务并行转换 - 进度查看脚本
# ========================================

CHECKPOINT_DIR="${1:-/shared/storage/checkpoints}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "错误: Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

echo "=========================================="
echo "任务进度总览"
echo "=========================================="
echo "Checkpoint目录: $CHECKPOINT_DIR"
echo ""

# 统计各状态的任务数
completed=0
processing=0
failed=0
total=0

# 遍历所有task checkpoint文件
for checkpoint in "$CHECKPOINT_DIR"/task_*.json; do
    if [ -f "$checkpoint" ]; then
        total=$((total + 1))
        task_id=$(basename "$checkpoint" | sed 's/task_//' | sed 's/.json//')
        status=$(jq -r '.status // "unknown"' "$checkpoint" 2>/dev/null || echo "unknown")
        
        case "$status" in
            completed)
                completed=$((completed + 1))
                ;;
            processing)
                processing=$((processing + 1))
                ;;
            failed)
                failed=$((failed + 1))
                ;;
        esac
    fi
done

# 显示统计
echo "总任务数: $total"
echo "已完成: $completed"
echo "处理中: $processing"
echo "失败: $failed"
echo ""

# 显示详细信息
echo "=========================================="
echo "任务详情"
echo "=========================================="

# 按状态分组显示
echo ""
echo "【处理中的任务】"
for checkpoint in "$CHECKPOINT_DIR"/task_*.json; do
    if [ -f "$checkpoint" ]; then
        status=$(jq -r '.status // "unknown"' "$checkpoint" 2>/dev/null || echo "unknown")
        if [ "$status" == "processing" ]; then
            task_id=$(basename "$checkpoint" | sed 's/task_//' | sed 's/.json//')
            pid=$(jq -r '.pid // "N/A"' "$checkpoint" 2>/dev/null || echo "N/A")
            timestamp=$(jq -r '.timestamp // 0' "$checkpoint" 2>/dev/null || echo "0")
            elapsed=$(($(date +%s) - ${timestamp%.*}))
            echo "  Task $task_id (PID: $pid, 运行时间: ${elapsed}s)"
        fi
    fi
done

echo ""
echo "【已完成的任务】"
for checkpoint in "$CHECKPOINT_DIR"/task_*.json; do
    if [ -f "$checkpoint" ]; then
        status=$(jq -r '.status // "unknown"' "$checkpoint" 2>/dev/null || echo "unknown")
        if [ "$status" == "completed" ]; then
            task_id=$(basename "$checkpoint" | sed 's/task_//' | sed 's/.json//')
            num_episodes=$(jq -r '.num_episodes // "N/A"' "$checkpoint" 2>/dev/null || echo "N/A")
            echo "  Task $task_id ($num_episodes episodes)"
        fi
    fi
done

echo ""
echo "【失败的任务】"
for checkpoint in "$CHECKPOINT_DIR"/task_*.json; do
    if [ -f "$checkpoint" ]; then
        status=$(jq -r '.status // "unknown"' "$checkpoint" 2>/dev/null || echo "unknown")
        if [ "$status" == "failed" ]; then
            task_id=$(basename "$checkpoint" | sed 's/task_//' | sed 's/.json//')
            error=$(jq -r '.error // "Unknown error"' "$checkpoint" 2>/dev/null || echo "Unknown error")
            echo "  Task $task_id"
            echo "    错误: $error"
        fi
    fi
done

# 显示活跃的锁
echo ""
echo "=========================================="
echo "活跃的锁"
echo "=========================================="
lock_count=0
if [ -d "$CHECKPOINT_DIR/task_locks" ]; then
    for lockfile in "$CHECKPOINT_DIR/task_locks"/task_*.lock; do
        if [ -f "$lockfile" ]; then
            # 检查lockfile是否被占用
            if fuser "$lockfile" 2>/dev/null | grep -q .; then
                lock_count=$((lock_count + 1))
                task_id=$(basename "$lockfile" | sed 's/task_//' | sed 's/.lock//')
                echo "  Task $task_id (锁定中)"
            fi
        fi
    done
fi

if [ $lock_count -eq 0 ]; then
    echo "  无活跃的锁"
fi

echo ""
echo "=========================================="
