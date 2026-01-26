# 快速修复指南

## 一键修复已转换的数据

如果你的数据已经转换完成，但 episode 索引有间隙，按以下步骤操作：

### 步骤 1：预览（查看哪些需要修复）

```bash
python fix_episode_indices.py --dataset-path /path/to/your/output --dry-run
```

这会显示：
- 哪些任务有间隙
- 具体的索引映射
- 但**不会**做任何修改

### 步骤 2：修复所有任务

```bash
python fix_episode_indices.py --dataset-path /path/to/your/output
```

脚本会：
1. 自动创建备份（`task_XXX_backup`）
2. 要求你确认（输入 `yes`）
3. 重命名所有文件
4. 更新所有元数据

### 步骤 3：验证结果

检查文件是否连续：

```bash
# 检查 parquet 文件
ls /path/to/your/output/task_355/data/chunk-000/

# 应该看到：
# episode_000000.parquet
# episode_000001.parquet
# episode_000002.parquet
# episode_000003.parquet
# ...（连续，无间隙）
```

## 常见场景

### 场景 1：只修复一个任务

```bash
python fix_episode_indices.py --dataset-path output --task-id 355
```

### 场景 2：批量修复所有任务

```bash
python fix_episode_indices.py --dataset-path output
```

### 场景 3：先测试一个，再批量

```bash
# 1. 先测试一个任务
python fix_episode_indices.py --dataset-path output --task-id 355

# 2. 验证结果正确后，修复其他任务
python fix_episode_indices.py --dataset-path output
```

## 示例输出

```
================================================================================
Episode Index Gap Fixer
================================================================================
Dataset path: output

Found 5 task(s) to process

================================================================================
Processing task: task_355
================================================================================
Found 10 episodes
Existing indices: [0, 1, 3, 5, 6, 7, 9, 11, 13, 15]
Expected indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
⚠ Gaps detected in episode indices!

Index mapping:
  Episode 3 -> 2
  Episode 5 -> 3
  Episode 6 -> 4
  Episode 7 -> 5
  Episode 9 -> 6
  Episode 11 -> 7
  Episode 13 -> 8
  Episode 15 -> 9

Proceed with fixing episode indices? (yes/no): yes

Creating backup: output/task_355_backup
✓ Backup created

Renaming episode files...
✓ All files renamed

Updating metadata files...
✓ All metadata updated

================================================================================
✓ Episode indices fixed successfully!
================================================================================
```

## 安全提示

✅ **自动备份** - 脚本会自动创建备份，出问题可以恢复  
✅ **确认提示** - 修改前会要求确认  
✅ **预览模式** - 可以先用 `--dry-run` 查看  

## 如果出错了怎么办？

恢复备份：

```bash
# 删除修复后的数据
rm -rf output/task_355

# 恢复备份
mv output/task_355_backup output/task_355
```

## 完成！

修复后，你的数据集 episode 索引将是连续的：
- episode_000000
- episode_000001
- episode_000002
- ...

没有间隙，可以正常使用了！
