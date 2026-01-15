# AgiBotWorld 数据转换工具 - Deployment Package

## 文件清单

```
deployment/
├── convert_multi_task.py          # 核心转换脚本（竞争式多任务）
├── convert_to_lerobot.py          # 原始转换脚本（被上面的脚本导入）
├── run.sh                         # 一键启动脚本 ⭐
├── check_multi_task_progress.sh   # 进度监控脚本
├── requirements.txt               # Python依赖
└── README.md                      # 使用说明
```

## 快速部署

### 1. 在共享存储上部署代码

```bash
# 将deployment文件夹复制到共享存储
scp -r deployment/ user@server:/shared/storage/AgiBotWorld-Alpha/
```

或者直接在共享存储上克隆：
```bash
ssh node1
cd /shared/storage
git clone <repo-url> AgiBotWorld-Alpha
cd AgiBotWorld-Alpha/deployment
```

### 2. 安装依赖（每个节点都需要）

```bash
pip install -r requirements.txt
```

### 3. 配置路径

编辑 `run.sh`，修改前3行：
```bash
SRC_PATH="/shared/storage/source_data"      # 你的源数据路径
TGT_PATH="/shared/storage/lerobot_data"     # 你的目标路径
CHECKPOINT_DIR="/shared/storage/checkpoints" # checkpoint目录（必须共享）
```

### 4. 在所有节点运行相同命令

**节点1**:
```bash
cd /shared/storage/AgiBotWorld-Alpha/deployment
bash run.sh
```

**节点2** (完全相同):
```bash
cd /shared/storage/AgiBotWorld-Alpha/deployment
bash run.sh
```

**节点3, 4, 5...** (都运行相同命令)

### 5. 监控进度

在任意节点：
```bash
cd /shared/storage/AgiBotWorld-Alpha/deployment
bash check_multi_task_progress.sh /shared/storage/checkpoints
```

## 工作原理

**竞争式处理**：
- 所有节点运行完全相同的命令
- 每个节点启动8个进程
- 每个进程尝试获取task锁，成功则处理，失败则跳过
- 自动负载均衡，快的节点处理更多tasks

```
节点1 (8进程) ──┐
节点2 (8进程) ──┼──> 都尝试处理 [task_327, 328, 329, ...]
节点3 (8进程) ──┘
                   ↓
           通过task锁自动协调，避免重复处理
```

## 数据结构要求

源数据结构：
```
/shared/storage/source_data/
├── observations/
│   ├── 327/              # Task 327
│   │   ├── 0001/         # Episode 1
│   │   │   ├── videos/
│   │   │   └── depth/
│   │   ├── 0002/         # Episode 2
│   │   └── ...
│   ├── 328/              # Task 328
│   └── ...
├── task_info/
│   ├── task_327.json
│   ├── task_328.json
│   └── ...
└── proprio_stats/
    ├── 327/
    │   ├── 0001/
    │   │   └── proprio_stats.h5
    │   └── ...
    └── ...
```

输出结构：
```
/shared/storage/lerobot_data/
└── agibotworld/
    ├── task_327/
    │   ├── data/
    │   │   ├── episode_000000.parquet
    │   │   └── ...
    │   ├── videos/
    │   │   └── observation.images.*/
    │   │       └── episode_*.mp4
    │   ├── meta_info/
    │   └── stats/
    └── task_328/
        └── ...
```

## 参数调优

编辑 `run.sh` 中的并行参数：

```bash
NUM_PROCESSES=8          # 每节点进程数（推荐8-16）
NUM_WORKERS_PER_TASK=4   # 每task内workers（推荐2-4）
```

**调优建议**：
- CPU密集：增加 `NUM_PROCESSES`，减少 `NUM_WORKERS_PER_TASK`
- 内存受限：都减小
- I/O密集：保持默认

## 常见问题

### Q: 可以中途停止吗？
A: 可以！按Ctrl+C停止，下次运行会自动断点续传

### Q: 某个节点挂了怎么办？
A: 重启运行相同命令即可，已完成的自动跳过

### Q: 如何确认全部完成？
A: 运行进度监控脚本，检查 completed 数量是否等于总task数

### Q: 可以动态增减节点吗？
A: 完全可以！随时启动/停止节点，通过锁自动协调

## 手动运行（不用run.sh）

如果想手动控制参数：

```bash
cd /shared/storage/AgiBotWorld-Alpha/deployment

python convert_multi_task.py \
  --src_path /shared/storage/source_data \
  --tgt_path /shared/storage/lerobot_data \
  --checkpoint_dir /shared/storage/checkpoints \
  --num_processes 8 \
  --num_workers_per_task 4
```

## 故障恢复

### 删除过期锁文件
```bash
# 删除1小时前的锁（说明进程已死）
find /shared/storage/checkpoints/task_locks -name "*.lock" -mmin +60 -delete
```

### 重新处理失败的task
```bash
# 删除失败task的checkpoint
rm /shared/storage/checkpoints/task_327.json
rm /shared/storage/checkpoints/task_locks/task_327.lock

# 重新运行
bash run.sh
```

### 重置所有进度
```bash
# 谨慎操作！会删除所有checkpoint
rm -rf /shared/storage/checkpoints/*
bash run.sh
```

## 性能预估

假设：
- 100个tasks
- 每个task 1000个episodes
- 4个节点，每节点8进程

预计：
- 并发度: 4节点 × 8进程 = 32个tasks同时处理
- 完成时间: 取决于单个task处理时间

## 技术支持

遇到问题请检查：
1. 所有节点能访问共享存储（`ls /shared/storage`）
2. Python依赖已安装（`python -c "import lerobot"`）
3. 源数据结构正确（`ls /shared/storage/source_data/task_info`）
4. Checkpoint目录可写（`touch /shared/storage/checkpoints/test`）
