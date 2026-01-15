#!/bin/bash

# ========================================
# 一键启动脚本 - 所有节点运行此脚本
# ========================================

# 配置路径（根据实际情况修改）
SRC_PATH="/shared/storage/source_data"
TGT_PATH="/shared/storage/lerobot_data"
CHECKPOINT_DIR="/shared/storage/checkpoints"

# 并行参数
NUM_PROCESSES=8          # 每个节点的进程数
NUM_WORKERS_PER_TASK=4   # 每个task内的并行worker数

# ========================================
# 检查环境
# ========================================

# 检查Python和依赖
if ! command -v python &> /dev/null; then
    echo "错误: 未找到python命令"
    exit 1
fi

echo "检查依赖包..."
python -c "import lerobot, h5py, torch, numpy, PIL, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少依赖包，请运行: pip install -r requirements.txt"
    exit 1
fi

# 检查路径
if [ ! -d "$SRC_PATH" ]; then
    echo "错误: 源数据路径不存在: $SRC_PATH"
    echo "请修改脚本中的 SRC_PATH"
    exit 1
fi

if [ ! -d "$SRC_PATH/task_info" ]; then
    echo "错误: task_info目录不存在: $SRC_PATH/task_info"
    exit 1
fi

# 创建必要目录
mkdir -p "$TGT_PATH"
mkdir -p "$CHECKPOINT_DIR"

# ========================================
# 显示配置
# ========================================

echo ""
echo "=========================================="
echo "多任务并行转换 - 竞争式处理"
echo "=========================================="
echo "主机: $(hostname)"
echo "PID: $$"
echo "源路径: $SRC_PATH"
echo "目标路径: $TGT_PATH"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "进程数: $NUM_PROCESSES"
echo "每task的workers: $NUM_WORKERS_PER_TASK"
echo "=========================================="
echo ""
echo "所有节点运行相同的命令，通过task锁自动协调"
echo "按Ctrl+C可以随时中断（支持断点续传）"
echo ""
echo "启动中..."
sleep 2

# ========================================
# 运行转换
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/convert_multi_task.py" \
    --src_path "$SRC_PATH" \
    --tgt_path "$TGT_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --num_processes "$NUM_PROCESSES" \
    --num_workers_per_task "$NUM_WORKERS_PER_TASK"

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ 节点完成!"
else
    echo "✗ 节点退出 (exit code: $exit_code)"
fi
echo "=========================================="

exit $exit_code
