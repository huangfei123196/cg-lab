# 计算机图形学实验一：图形学开发工具

本项目完成了计算机图形学实验一中的开发环境搭建与测试任务：使用 `uv` 管理 Python 项目环境，采用 `src layout` 组织代码，并基于 `Taichi` 实现一个由鼠标位置吸引的万有引力粒子群仿真。程序会优先尝试使用 GPU 后端执行并行计算，运行时可通过终端输出查看 Taichi 实际启动的计算架构。

## 一、项目架构

项目采用课程要求的 `src/Work0` 模块化布局，将配置、计算逻辑和程序入口分离，避免把所有代码堆在根目录中。

```text
CG-Lab/
├── pyproject.toml
├── uv.lock
├── README.md
├── assets/
│   └── demo.gif
├── src/
│   └── Work0/
│       ├── __init__.py
│       ├── config.py
│       ├── physics.py
│       └── main.py
└── tools/
    └── render_demo.py
```

各文件作用如下：

| 文件 | 作用 |
| --- | --- |
| `pyproject.toml` | uv 项目配置文件，记录 Python 版本和依赖包 |
| `uv.lock` | 锁定依赖版本，保证不同设备上安装结果一致 |
| `src/Work0/config.py` | 集中保存粒子数量、引力强度、阻尼系数、窗口大小和颜色等参数 |
| `src/Work0/physics.py` | 定义 Taichi field 和 GPU 并行 kernel，负责粒子物理计算 |
| `src/Work0/main.py` | 程序入口，负责初始化 Taichi、读取鼠标输入并绘制窗口 |
| `tools/render_demo.py` | 离屏生成 README 中使用的演示 GIF |

## 二、代码逻辑

程序整体分为三层：

1. 参数配置层：`config.py` 保存可调参数，例如 `NUM_PARTICLES`、`GRAVITY_STRENGTH`、`DRAG_COEF` 和 `WINDOW_RES`。如果需要降低粒子数量或调整显示效果，只需要修改配置文件或设置环境变量。
2. 物理计算层：`physics.py` 使用 Taichi 的 `Vector.field` 保存粒子位置 `pos` 和速度 `vel`。`init_particles()` 负责随机初始化粒子，`update_particles()` 通过 Taichi kernel 并行更新所有粒子的运动状态。
3. 交互渲染层：`main.py` 调用 `ti.init(arch=ti.gpu)` 优先启用 GPU 后端，通过 `ti.GUI` 创建窗口，循环读取鼠标位置，并将更新后的粒子位置绘制到屏幕上。

核心运动逻辑是：每个粒子都会受到鼠标位置的吸引，程序根据粒子到鼠标的方向和距离计算加速度；随后更新速度和位置，并加入阻尼系数模拟能量损耗。当粒子碰到窗口边界时，会进行反弹处理，使粒子始终保持在可视区域内。

## 三、实现功能

- 使用 `uv` 创建项目级虚拟环境，避免污染全局 Python 环境。
- 使用 `src/Work0` 布局组织实验代码，体现配置、计算和渲染分层。
- 使用 Taichi field 保存一万个粒子的状态。
- 使用 Taichi kernel 并行执行粒子初始化和物理更新。
- 鼠标移动时，粒子群会实时向鼠标位置聚集。
- 粒子碰到边界后会反弹，并受到阻尼影响逐渐稳定。
- README 中提供 GIF 动图，便于在 GitHub 或 Gitee 仓库页面直接查看效果。

## 四、运行方法

在项目根目录执行：

```powershell
uv sync
uv run -m src.Work0.main
```

运行后会弹出名为 `Experiment 0: Taichi Gravity Swarm` 的窗口。移动鼠标时，粒子群会被鼠标位置吸引，并在窗口边界发生反弹。

如果电脑比较卡，可以临时降低粒子数量：

```powershell
$env:CG_LAB_NUM_PARTICLES="3000"
uv run -m src.Work0.main
```

## 五、运行效果

![粒子群演示](assets/demo.gif)

本机测试时，Taichi 成功启动了 CUDA 后端，说明 GPU 已经接管计算：

```text
[Taichi] Starting on arch=cuda
```

在其他设备上，如果终端中出现类似以下输出，也说明 Taichi 成功启用了 GPU 或图形后端：

```text
[Taichi] Starting on architecture: cuda
[Taichi] Starting on architecture: vulkan
[Taichi] Starting on architecture: opengl
```

如果输出为 `cpu`，程序仍然可以运行，只是粒子数量较大时帧率会下降。

## 六、生成 README 演示 GIF

仓库里提供了一个离屏渲染脚本，可重新生成 `assets/demo.gif`：

```powershell
uv run python tools/render_demo.py
```

该脚本复用同一套 Taichi 物理 kernel，只是把每一帧的位置数据绘制到 GIF 中，方便在 GitHub 或 Gitee README 中直接展示运行效果。
