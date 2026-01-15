# 竞争式多任务并行转换 - 快速开始

## 核心特性

**所有节点运行完全相同的命令**，通过task锁自动竞争处理：
- ✅ 无需手动分配节点编号
- ✅ 每个节点自动启动8个进程
- ✅ Task级别文件锁自动协调
- ✅ 代码放在共享存储，所有节点访问同一份代码

## 使用步骤

### 1. 将代码部署到共享存储

```bash
# 在任意一个节点上
cd /shared/storage
git clone <repo> AgiBotWorld-Alpha
cd AgiBotWorld-Alpha
pip install lerobot h5py torch numpy pillow tqdm
```

### 2. 在所有节点运行相同命令

**节点1**:
```bash
cd /shared/storage/AgiBotWorld-Alpha

python scripts/convert_multi_task.py \
  --src_path /shared/storage/source_data \
  --tgt_path /shared/storage/lerobot_data \
  --checkpoint_dir /shared/storage/checkpoints \
  --num_processes 8
```

**节点2** (完全相同的命令):
```bash
cd /shared/storage/AgiBotWorld-Alpha

python scripts/convert_multi_task.py \
  --src_path /shared/storage/source_data \
  --tgt_path /shared/storage/lerobot_data \
  --checkpoint_dir /shared/storage/checkpoints \
  --num_processes 8
```

**节点3, 4, 5...** (都是相同的命令)

### 3. 监控进度

```bash
bash scripts/check_multi_task_progress.sh /shared/storage/checkpoints
```

## 工作原理

```
共享存储: /shared/storage/checkpoints/task_locks/
├── task_327.lock  ← 节点1的进程A获取
├── task_328.lock  ← 节点2的进程B获取
├── task_329.lock  ← 节点1的进程C获取
└── ...

所有节点处理相同的task列表: [327, 328, 329, ...]

节点1 (8个进程):
├── 进程1 → 尝试task_327 ✅ 获得锁 → 处理
├── 进程2 → 尝试task_328 ❌ 已被锁 → 跳过 → 尝试task_329 ✅ → 处理
├── 进程3 → 尝试task_327 ❌ 已被锁 → 跳过 → 尝试task_330 ✅ → 处理
└── ...

节点2 (8个进程):
├── 进程1 → 尝试task_327 ❌ 已被锁 → 跳过 → 尝试task_328 ✅ → 处理
├── 进程2 → 尝试task_329 ❌ 已被锁 → 跳过 → 尝试task_331 ✅ → 处理
└── ...
```

**关键机制**:
- 每个task只能被一个进程获取锁
- 获取失败的进程自动跳过该task
- 已完成的task会被所有进程跳过

## 完整示例

假设有4个计算节点，100个tasks：

```bash
# 所有节点执行相同命令（可以用脚本批量执行）
for node in node1 node2 node3 node4; do
  ssh $node "cd /shared/storage/AgiBotWorld-Alpha && \
    python scripts/convert_multi_task.py \
      --src_path /shared/storage/source_data \
      --tgt_path /shared/storage/lerobot_data \
      --checkpoint_dir /shared/storage/checkpoints \
      --num_processes 8 &"
done
```

或者在每个节点手动执行（推荐用tmux/screen）：
```bash
# 节点1
tmux new -s convert
cd /shared/storage/AgiBotWorld-Alpha
python scripts/convert_multi_task.py \
  --src_path /shared/storage/source_data \
  --tgt_path /shared/storage/lerobot_data \
  --checkpoint_dir /shared/storage/checkpoints

# 节点2 (相同命令)
tmux new -s convert
cd /shared/storage/AgiBotWorld-Alpha
python scripts/convert_multi_task.py \
  --src_path /shared/storage/source_data \
  --tgt_path /shared/storage/lerobot_data \
  --checkpoint_dir /shared/storage/checkpoints

# 节点3, 4... (相同命令)
```

## 参数说明

```bash
python scripts/convert_multi_task.py \
  --src_path <源数据路径> \                    # 必需
  --tgt_path <目标数据路径> \                  # 必需
  --checkpoint_dir <checkpoint目录> \          # 必需，必须是共享存储
  --num_processes 8 \                          # 可选，每节点进程数（默认8）
  --num_workers_per_task 4 \                   # 可选，每task内workers（默认4）
  --task_ids 327 328 329                       # 可选，指定处理哪些tasks
```

## 常见问题

### Q: 为什么要把代码放在共享存储？
A: 不是必须，但推荐这样做：
- 方便：所有节点用同一份代码，更新方便
- 一致性：避免不同节点代码版本不一致

如果不想放共享存储：
```bash
# 每个节点本地部署
cd /local/path
git clone <repo>
# 但要确保所有节点代码版本一致
```

### Q: 可以动态增加/减少节点吗？
A: **完全可以**！随时启动/停止节点：
```bash
# 新增节点5 - 运行相同命令即可
ssh node5 "cd /shared/storage/AgiBotWorld-Alpha && python scripts/convert_multi_task.py ..."

# 已完成的tasks会自动跳过
# 未完成的tasks会被新节点处理
```

### Q: 某个节点挂了怎么办？
A: 
1. 该节点正在处理的tasks会留下锁文件
2. 锁文件1小时后自动过期
3. 或者手动删除锁文件：
```bash
# 查看哪些锁过期了
find /shared/storage/checkpoints/task_locks -name "*.lock" -mmin +60

# 删除过期的锁
find /shared/storage/checkpoints/task_locks -name "*.lock" -mmin +60 -delete

# 重启节点，会自动重新处理未完成的tasks
```

### Q: 如何确认所有tasks都完成了？
A: 
```bash
bash scripts/check_multi_task_progress.sh /shared/storage/checkpoints

# 或者检查
cd /shared/storage/checkpoints
total_tasks=$(ls task_*.json 2>/dev/null | wc -l)
completed=$(grep -l '"status": "completed"' task_*.json 2>/dev/null | wc -l)
echo "完成: $completed / $total_tasks"
```

## 性能优化

### CPU密集场景
```bash
# 增加进程数（如果CPU核心多）
--num_processes 16
--num_workers_per_task 2
```

### 内存受限场景
```bash
# 减少并发
--num_processes 4
--num_workers_per_task 2
```

### I/O密集场景
```bash
# 保持默认
--num_processes 8
--num_workers_per_task 4
```

## 对比旧方案

**旧方案** (需要手动配置):
```bash
# 节点1
SHARD_INDEX=0 TOTAL_SHARDS=4 bash run_node.sh

# 节点2
SHARD_INDEX=1 TOTAL_SHARDS=4 bash run_node.sh

# 节点3
SHARD_INDEX=2 TOTAL_SHARDS=4 bash run_node.sh

# 节点4
SHARD_INDEX=3 TOTAL_SHARDS=4 bash run_node.sh
```

**新方案** (所有节点相同命令):
```bash
# 所有节点都执行
python scripts/convert_multi_task.py \
  --src_path /shared/storage/source_data \
  --tgt_path /shared/storage/lerobot_data \
  --checkpoint_dir /shared/storage/checkpoints
```

优势：
- ✅ 无需手动分配节点编号
- ✅ 动态增减节点更容易
- ✅ 自动负载均衡（快的节点处理更多tasks）
- ✅ 更简单的运维
