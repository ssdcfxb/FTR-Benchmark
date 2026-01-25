# FTR-Benchmark 项目结构说明

本文件汇总了项目主要模块的结构与功能说明，包括环境库 (`ftr_envs`)、算法库 (`ftr_algo`) 和执行脚本 (`scripts`)。

---

## 1. ftr_envs - 环境库

`ftr_envs` 是基于 Isaac Lab 构建的 FTR (Flipper Track Robot) 强化学习环境的核心组件库。

### 📁 目录概览

```
ftr_envs/
├── assets/          # 机器人定义、物理模型与地形资源
├── tasks/           # RL 任务逻辑、环境实现与算法配置
├── envs/            # 环境功能增强（如指标收集）
└── utils/           # 通用工具函数库
```

### 1.1 assets - 资产与配置模块

管理机器人及环境的物理与视觉资产。

| 子目录/文件 | 文件名示例 | 作用描述 |
|:---|:---|:---|
| **`ftr.py`** | - | **配置入口**。定义 `FTR_CFG` (机器人组件、传感器) 和 `FTR_SIM_CFG` (物理引擎参数)。 |
| **`articulation/`** | `ftr.py` | **运动学控制**。封装底盘轮子与鳍状肢关节的低级控制接口 (位置/速度控制)。 |
| **`terrain/`** | `terrain.py` | **地形管理类**。负责解析配置、加载 USD 场景及生成高度图观测。 |
| &emsp;`birth/` | `*.json` | **任务点位**。定义每种地形的机器人出生点 (start) 和目标点 (target) 列表。 |
| &emsp;`config/` | `*.yaml` | **地形参数**。定义地形的物理属性、障碍物位置及渲染设置。 |
| &emsp;`usd/` | `*.usd` | **3D 模型**。存储机器人 (`ftr/`) 和各种地形场景的 USD 格式文件。 |
| &emsp;`map/` | `*.map` | **高度图数据**。用于计算机器人对地面的感知观测。 |

### 1.2 tasks - 任务定义模块

包含具体的强化学习环境逻辑和算法配置。

#### 核心任务：Crossing (越障)
机器人在复杂地形（楼梯、废墟等）上自主导航。

| 类别 | 文件 | 作用描述 |
|:---|:---|:---|
| **环境实现** | `ftr_env.py` | **环境基类**。处理通用的场景初始化、物理仿真步进、基础状态更新。 |
| | `crossing_env.py` | **Crossing 任务类**。实现特定的奖励函数 (距离、姿态惩罚)、终止条件 (翻车、超时) 和观测空间。 |
| **算法配置** | `agents/*.yaml` | **超参数文件**。包含不同 RL 算法的训练参数（网络结构、学习率等）。<br>• 单智能体: `*_ppo`, `*_ddpg`, `*_sac`, `*_td3`<br>• 多智能体: `*_mappo`, `*_happo`, `*_ippo` |

#### 扩展任务
| 任务名 | 核心文件 | 描述 |
|:---|:---|:---|
| `prey` | `prey_env.py` | **多智能体追捕**。多个 FTR 机器人协作捕获目标。 |
| `push_cube` | `push_cube_env.py` | **协同推箱**。机器人通过接触力推动物体移动。 |
| `trans_cargo` | `trans_cargo_env.py` | **货物运输**。模拟负载状态下的运动控制。 |
| `anymal_d` | `anymal_d_cfg.py` | **四足对比**。引入 ANYmal 机器人作为基准对比任务。 |

### 1.3 envs - 环境增强模块

提供环境的包装器 (Wrapper) 以扩展功能。

| 文件 | 作用描述 |
|:---|:---|
| **`metrics_env.py`** | **指标评估包装器**。在训练/测试循环中自动收集性能数据（成功率、翻车率、轨迹长度等），支持数据持久化 (CSV/Pickle)。 |

### 1.4 utils - 工具模块

通用辅助函数库。

| 文件 | 作用描述 |
|:---|:---|
| **`torch.py`** | **张量操作**。提供 PyTorch 辅助函数，如高斯噪声注入 (`add_noise`)、随机数生成等。 |
| **`omega_conf.py`** | **配置解析**。扩展 OmegaConf 功能，支持配置文件中的自定义运算（加减乘除、逻辑判断）。 |
| **`prim.py`** | **USD 操作**。Omniverse 场景操作工具，用于动态创建 Prim、设置材质属性、物理摩擦力及可见性。 |

---

## 2. ftr_algo - 算法库

`ftr_algo` 是本项目原生的强化学习算法库，包含单智能体 (SARL) 和多智能体 (MARL) 的算法实现及调度器。

### 📁 目录概览

```
ftr_algo/
├── algorithms/      # RL 算法核心实现 (SARL, MARL)
├── utils/           # 训练流程控制与通用工具
├── executor.py      # 算法注册与执行入口
└── http_deploy.py   # 模型部署接口
```

### 2.1 algorithms - 算法实现

管理各种 RL 算法的策略、网络结构和训练器。

| 模块 | 子目录/文件 | 作用描述 |
|:---|:---|:---|
| **rl (单智能体)** | `ppo/`, `ddpg/`, `sac/`, `td3/`, `trpo/` | **SARL 算法实现**。每个子目录包含：<br>• `*.py`: 核心训练逻辑 (Trainer)<br>• `module.py`: Actor-Critic 网络结构<br>• `storage.py`: 经验回放 buffer 实现 |
| **marl (多智能体)** | `runner.py` | MAPPO/HAPPO 等算法的共用运行器 (Runner)。 |
| | `*_trainer.py` | 算法的訓練器 (Trainer)，如 `mappo_trainer.py`。 |
| | `*_policy.py` | 算法的策略网络 (Policy)，如 `mappo_policy.py`。 |
| | `maddpg/` | MADDPG 算法的独立实现包。 |

### 2.2 utils - 流程控制与工具

负责训练循环的管理、日志记录和数据处理。

| 文件 | 作用描述 |
|:---|:---|
| **核心流程** | |
| `process_sarl.py` | **SARL 训练流程**。负责初始化 SARL 算法 (PPO, SAC 等) 并启动训练循环。 |
| `process_marl.py` | **MARL 训练流程**。配置多智能体环境参数，初始化 Runner 并控制训练过程。 |
| **辅助工具** | |
| `logger/` | 包含 `plotter.py` 和 `tools.py`，用于 TensorBoard 日志记录和绘图。 |
| `util.py` | 通用工具函数。 |

### 2.3 根目录文件

| 文件 | 作用描述 |
|:---|:---|
| **`executor.py`** | **执行器**。注册并管理所有支持的算法 (AlgoFactory)，提供统一的算法实例化接口。 |
| **`http_deploy.py`** | **部署服务**。提供 HTTP 接口，用于将训练好的模型部署为推理服务。 |

---

## 3. scripts - 脚本库

`scripts` 目录包含用于训练、评估和调试强化学习智能体的执行脚本，支持多种 RL 框架。

### 📁 目录概览

```
scripts/
├── debug.py           # 环境调试与数据分析
├── ftr_algo/          # FTR 专用算法框架
├── rl_games/          # RL-Games 框架接口
└── skrl/              # SKRL 框架接口
```

### 3.1 根目录脚本

| 文件 | 作用描述 |
|:---|:---|
| **`debug.py`** | **调试与验证**。启动 Isaac Sim 环境而不进行训练，用于：<br>• 检查环境配置和渲染效果<br>• 收集并可视化传感器数据（使用 matplotlib/pandas）<br>• 验证环境复位和物理逻辑 |

### 3.2 ftr_algo - 专用算法脚本

基于项目自带的算法实现（ftr_algo 包）进行训练。

| 文件 | 作用描述 |
|:---|:---|
| **`train.py`** | **核心训练脚本**。支持单智能体 (SARL) 和多智能体 (MARL) 算法。<br>• 自动根据算法类型选择 Env Wrapper (SARLWrap/MARLWrap)<br>• 支持算法：PPO, DDPG, SAC, MAPPO, HAPPO 等 |

### 3.3 外部框架集成脚本

提供了主流 RL 库的适配接口。

#### rl_games/
| 文件 | 作用描述 |
|:---|:---|
| **`train.py`** | 使用 **RL-Games** 库启动训练。 |
| **`play.py`** | 加载 RL-Games 训练的 checkpoint 进行推理/演示。 |

#### skrl/
| 文件 | 作用描述 |
|:---|:---|
| **`train.py`** | 使用 **SKRL** 库启动训练。支持 PyTorch 和 JAX 后端 (IPPO, MAPPO)。 |
| **`play.py`** | 加载 SKRL 训练的 checkpoint 进行推理/演示。 |
