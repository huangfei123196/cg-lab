# 计算机图形学实验一：图形学开发工具

本项目使用 `uv + src layout + Taichi` 搭建一个隔离、可复现的图形学实验工程，并实现一个由鼠标吸引的万有引力粒子群仿真。程序会优先尝试使用 GPU 后端，运行时可以在终端中看到 Taichi 的实际启动架构。

## 项目结构

```text
CG-Lab/
├── pyproject.toml
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

`src/Work0/config.py` 负责集中管理粒子数量、引力、阻尼、窗口尺寸和颜色等参数。`src/Work0/physics.py` 只保存 Taichi field 和 GPU 并行 kernel，负责粒子初始化、引力计算、速度更新和边界反弹。`src/Work0/main.py` 是程序入口，完成 Taichi 初始化、鼠标输入读取和 GUI 渲染。

## 运行方法

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

## 运行效果

![粒子群演示](assets/demo.gif)

终端中若出现类似以下输出，说明 Taichi 成功启用了 GPU 或图形后端：

```text
[Taichi] Starting on architecture: cuda
[Taichi] Starting on architecture: vulkan
[Taichi] Starting on architecture: opengl
```

如果输出为 `cpu`，程序仍然可以运行，只是粒子数量较大时帧率会下降。

## 生成 README 演示 GIF

仓库里提供了一个离屏渲染脚本，可重新生成 `assets/demo.gif`：

```powershell
uv run python tools/render_demo.py
```

该脚本复用同一套 Taichi 物理 kernel，只是把每一帧的位置数据绘制到 GIF 中，方便在 GitHub 或 Gitee README 中直接展示运行效果。

