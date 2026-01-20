#!/bin/bash

# ========================================
# 检查和修复不完整的转换任务
# ========================================

CHECKPOINT_DIR="${1:-/data/checkpoints}"
OUTPUT_DIR="${2:-/data/robocoin_data/agibotworld}"

echo "=========================================="
echo "检查转换状态"
echo "=========================================="
echo "Checkpoint目录: $CHECKPOINT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查jq是否可用
if ! command -v jq &> /dev/null; then
    echo "警告: jq未安装，使用python解析JSON"
    USE_PYTHON=1
else
    USE_PYTHON=0
fi

# JSON解析函数
get_json_field() {
    local file="$1"
    local field="$2"
    local default="$3"
    
    if [ $USE_PYTHON -eq 1 ]; then
        python3 -c "import json; f=open('$file'); d=json.load(f); print(d.get('$field', '$default'))" 2>/dev/null || echo "$default"
    else
        jq -r ".$field // \"$default\"" "$file" 2>/dev/null || echo "$default"
    fi
}

# 统计
total_tasks=0
completed_tasks=0
failed_tasks=0
incomplete_tasks=0
orphan_dirs=0

# 存储需要清理的任务
declare -a tasks_to_clean

# 检查所有checkpoint
echo "【1. 检查Checkpoint状态】"
echo "----------------------------------------"
for checkpoint in "$CHECKPOINT_DIR"/task_*.json; do
    if [ -f "$checkpoint" ]; then
        total_tasks=$((total_tasks + 1))
        task_id=$(basename "$checkpoint" | sed 's/task_//' | sed 's/.json//')
        status=$(get_json_field "$checkpoint" "status" "unknown")
        
        case "$status" in
            completed)
                completed_tasks=$((completed_tasks + 1))
                ;;
            failed)
                failed_tasks=$((failed_tasks + 1))
                echo "  ❌ Task $task_id - 失败"
                tasks_to_clean+=("$task_id")
                ;;
            processing)
                incomplete_tasks=$((incomplete_tasks + 1))
                echo "  ⚠️  Task $task_id - 处理中断（未完成）"
                tasks_to_clean+=("$task_id")
                ;;
            skipped)
                # 检查是否因为目录存在而跳过
                reason=$(get_json_field "$checkpoint" "reason" "")
                if [ "$reason" == "directory_already_exists" ]; then
                    # 检查目录是否真的完整
                    task_dir="$OUTPUT_DIR/task_$task_id"
                    if [ -d "$task_dir" ]; then
                        # 检查是否有data目录和文件
                        if [ ! -d "$task_dir/data" ] || [ -z "$(ls -A $task_dir/data 2>/dev/null)" ]; then
                            incomplete_tasks=$((incomplete_tasks + 1))
                            echo "  ⚠️  Task $task_id - 跳过但数据不完整"
                            tasks_to_clean+=("$task_id")
                        fi
                    fi
                fi
                ;;
            *)
                incomplete_tasks=$((incomplete_tasks + 1))
                echo "  ❓ Task $task_id - 状态未知: $status"
                tasks_to_clean+=("$task_id")
                ;;
        esac
    fi
done

# 检查孤立的输出目录（有目录但没有checkpoint）
echo ""
echo "【2. 检查孤立的输出目录】"
echo "----------------------------------------"
if [ -d "$OUTPUT_DIR" ]; then
    for task_dir in "$OUTPUT_DIR"/task_*; do
        if [ -d "$task_dir" ]; then
            task_id=$(basename "$task_dir" | sed 's/task_//')
            checkpoint_file="$CHECKPOINT_DIR/task_$task_id.json"
            
            if [ ! -f "$checkpoint_file" ]; then
                orphan_dirs=$((orphan_dirs + 1))
                echo "  🗑️  Task $task_id - 有目录但无checkpoint"
                tasks_to_clean+=("$task_id")
            fi
        fi
    done
fi

# 显示统计
echo ""
echo "=========================================="
echo "统计结果"
echo "=========================================="
echo "总任务数: $total_tasks"
echo "✅ 已完成: $completed_tasks"
echo "❌ 失败: $failed_tasks"
echo "⚠️  未完成: $incomplete_tasks"
echo "🗑️  孤立目录: $orphan_dirs"
echo "需要清理: ${#tasks_to_clean[@]}"
echo ""

# 如果有需要清理的任务
if [ ${#tasks_to_clean[@]} -gt 0 ]; then
    echo "=========================================="
    echo "需要清理的任务列表"
    echo "=========================================="
    for task_id in "${tasks_to_clean[@]}"; do
        echo "  - Task $task_id"
    done
    echo ""
    
    # 询问是否清理
    echo "是否清理这些不完整的任务？(y/N)"
    read -p "> " confirm
    
    if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
        echo ""
        echo "开始清理..."
        for task_id in "${tasks_to_clean[@]}"; do
            echo "  清理 Task $task_id..."
            
            # 删除输出目录
            task_dir="$OUTPUT_DIR/task_$task_id"
            if [ -d "$task_dir" ]; then
                rm -rf "$task_dir"
                echo "    ✓ 删除输出目录"
            fi
            
            # 删除checkpoint
            checkpoint_file="$CHECKPOINT_DIR/task_$task_id.json"
            if [ -f "$checkpoint_file" ]; then
                rm -f "$checkpoint_file"
                echo "    ✓ 删除checkpoint"
            fi
            
            # 删除锁文件
            lock_file="$CHECKPOINT_DIR/task_locks/task_$task_id.lock"
            if [ -f "$lock_file" ]; then
                rm -f "$lock_file"
                echo "    ✓ 删除锁文件"
            fi
        done
        
        echo ""
        echo "✅ 清理完成！"
        echo "可以重新运行转换脚本处理这些任务"
    else
        echo "已取消清理"
    fi
else
    echo "✅ 没有需要清理的任务"
fi

echo ""
echo "=========================================="
