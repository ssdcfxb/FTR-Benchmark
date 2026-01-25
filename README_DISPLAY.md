# 机器人状态采集脚本 (`scripts/display.py`) 使用说明

此脚本用于在多种地形上，通过物理模拟让机器人自由跌落并稳定，从而采集不同地形下的机器人稳定姿态数据（位置、姿态、关节角度等），用于基准测试或强化学习初始化。

## 1. 基础用法

```bash
python scripts/display.py [参数]
```

## 2. 参数详解

| 参数名称 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`--num_envs`** | int | `64` | **(重要)** 并行模拟的环境数量。数量越大，数据采集越快，但对 GPU 显存要求更高。 |
| **`--num_rounds`** | int | `10` | **(重要)** 数据采集的轮数。每一轮会根据 `scan_range` 对机器人的生成位置进行偏移，以覆盖更广泛的地形特征。 |
| **`--terrain`** | str | `"cur_mixed"` | **(重要)** 采样的地形类型。常见值：`cur_mixed` (混合地形), `cur_stairs_up` (上楼梯), `cur_rough` (崎岖路面) 等。 |
| **`--task`** | str | `"Ftr-Crossing-Direct-v0"` | 任务名称，通常不需要修改。 |
| **`--spawn_height`** | float | `0.2` | 机器人生成时距离地面的高度(米)。设得太高可能导致机器人翻车，太低可能导致穿模。 |
| **`--scan_range`** | float | `2.0` | **(重要)** 扫描范围(米)。在多轮采集 (`num_rounds > 1`) 中，机器人生成位置会在 `[-range/2, +range/2]` 之间移动，用于扫描不同地形切片。 |
| **`--robot_spacing`** | float | `2.0` | 机器人阵列中个体之间的网格间距(米)。 |
| **`--settle_steps`** | int | `100` | **(重要)** 每一轮等待机器人“稳定”的仿真步数。给物理引擎足够时间让机器人落地并静止。 |
| **`--sample_steps`** | int | `50` | 稳定之后，连续记录数据的步数。 |
| **`--output_dir`** | str | `"logs/data_collection"` | 数据保存目录。脚本会自动在该目录下创建带时间戳的子文件夹。 |

## 3. 重要参数使用示例

### 场景 1：快速测试脚本功能
仅运行 1 轮，使用 4 个环境，快速检查是否报错。
```bash
python scripts/display.py --num_envs 4 --num_rounds 1
```

### 场景 2：采集特定地形的大量数据
针对“上楼梯”地形，使用 256 个并行环境，采集 20 轮数据，增加机器人的生成高度防止卡住。
```bash
python scripts/display.py --terrain cur_stairs_up --num_envs 256 --num_rounds 20 --spawn_height 0.3
```

### 场景 3：大范围扫描地形
如果需要在更大范围的地形上采样（例如地形起伏较大），可以增大扫描范围 `scan_range`。
```bash
python scripts/display.py --num_rounds 50 --scan_range 10.0
```

### 场景 4：高精度稳定采样
如果发现机器人落地后还在晃动，可以增加 `settle_steps` 等待时间。
```bash
python scripts/display.py --settle_steps 500
```

## 4. 输出结果
脚本运行结束后，会在 `logs/data_collection/<地形名>_<时间戳>/` 目录下生成 `robot_states.csv` 文件，包含以下数据列：
*   `round`, `env_id`, `step`: 索引信息
*   `pos_x`, `pos_y`, `pos_z`: 机器人位置
*   `quat_0` ~ `quat_3`: 姿态四元数
*   `roll`, `pitch`, `yaw`: 欧拉角
*   `vel_x`... `ang_vel_x`...: 线速度与角速度
*   `joint_0`, `joint_1`...: 关节角度（摆臂/履带）
*   `stable_frames`: 机器人保持稳定的持续帧数
