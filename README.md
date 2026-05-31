# 计算机图形学实验一：图形学开发工具

## 一、项目架构

本项目使用 `uv` 管理 Python 环境，采用 `src/Work0` 作为实验代码目录，并将配置、物理计算和程序入口分离。

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

`config.py` 保存粒子数量、引力强度、阻尼系数、窗口大小和颜色等参数。`physics.py` 保存 Taichi field 与 GPU 并行计算 kernel。`main.py` 是程序入口，负责初始化 Taichi、读取鼠标位置并渲染粒子窗口。

## 二、代码逻辑

程序启动后先调用 `ti.init(arch=ti.gpu)`，优先使用 GPU 后端。随后 `init_particles()` 随机生成粒子初始位置和速度，`update_particles()` 在 Taichi kernel 中并行更新所有粒子。

每一帧中，程序读取鼠标坐标作为引力中心。粒子根据自身位置到鼠标位置的方向和距离计算加速度，再更新速度与位置；同时使用阻尼系数控制速度衰减，并在粒子碰到边界时进行反弹处理。最后，`main.py` 将 `pos` 中的粒子位置绘制到 GUI 窗口中。

## 三、实现功能

- 使用 `uv` 创建独立项目环境。
- 使用 `src/Work0` 布局组织实验代码。
- 使用 Taichi 管理一万个粒子的位置和速度。
- 使用 GPU 并行 kernel 完成粒子物理更新。
- 支持鼠标实时吸引粒子群。
- 支持窗口边界反弹与阻尼效果。

简单运行方式：

```powershell
uv sync
uv run -m src.Work0.main
```

## 四、效果展示

![粒子群演示](assets/demo.gif)

