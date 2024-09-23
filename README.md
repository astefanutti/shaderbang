# Shaderbang

Shaderbang is a Python toolkit for OpenGL rendering using [DRM/KMS](https://en.wikipedia.org/wiki/Direct_Rendering_Manager) on Linux.
It renders fullscreen without requiring any windowing system, like X or Wayland, making it ideal for embedded systems and headless environments.
It is compatible with shaders from [Shadertoy](https://www.shadertoy.com), and can be used as a library to build interactive applications.

It supports hot-pluggable input devices — keyboard, mouse, touchscreen, and trackpad — via [evdev](https://docs.kernel.org/input/input.html#evdev) for interactive rendering.
Shader files can be made directly executable with [shebangs](#shebangs), and Python scripts can embed their dependencies inline as [single-file scripts](#single-file-scripts) using [PEP 723](https://peps.python.org/pep-0723/).

It works with any GPU and display controller hardware, provided a DRM/KMS driver is available, from the [Raspberry Pi](https://ttt.io/glsl-raspberry-pi) and [Jetson Nano](https://ttt.io/glsl-jetson-nano) to NVIDIA RTX desktop GPUs.
More configurations are listed in the [compatibility](#compatibility) section.

In the following picture, a [Shadertoy shader](https://www.shadertoy.com/view/MsX3Wj) runs on the Raspberry Pi 4, connected to the official Raspberry Pi 7" touchscreen monitor, in WVGA resolution:

![A Shadertoy shader running on the Raspberry Pi 4](./raspberry_pi.jpg)

[Another shader](https://www.shadertoy.com/view/fstyD4) running on the Jetson Nano in full HD resolution:

![A Shadertoy shader running on the Jetson Nano](./jetson_nano.jpg)

## Installation

### System dependencies

Install the required DRM, GBM, EGL and OpenGL development libraries:

```shell
sudo apt install gcc libdrm-dev libgbm-dev libegl-dev libgles2-mesa-dev
```

Optionally, install X RandR for display lease support:

```shell
sudo apt install libxcb-randr0-dev
```

### Install

```shell
pip install shaderbang
```

### Install from source

For development, clone the repository and use an editable install:

```shell
pip install -e .
```

The `Makefile` can be used to quickly rebuild the native library during C development:

```shell
make
```

### Input device permissions

To handle input devices ([evdev](https://docs.kernel.org/input/input.html#evdev) events) without running as root:

```shell
sudo adduser $USER input
```

## Usage

Shaderbang provides the `shadertoy` command to run [Shadertoy](https://www.shadertoy.com) shaders:

```
shadertoy [-h] [--async-page-flip] [--atomic-drm-mode] [-C CONNECTOR]
          [-D DEVICE] [--mode MODE] [--refresh FREQ] [-n N]
          [-k UNIFORM] [--touchscreen UNIFORM] [--trackpad UNIFORM]
          [-c UNIFORM FILE] [-t UNIFORM FILE] [-v UNIFORM FILE]
          [-m <UNIFORM>.KEY VALUE]
          FILE
```

<details>
<summary>Arguments</summary>

```
  FILE                  the shadertoy file
  -h, --help            show this help message and exit
  --async-page-flip, --no-async-page-flip
                        use async page flipping
  --atomic-drm-mode, --no-atomic-drm-mode
                        use atomic mode setting
  -C CONNECTOR, --connector CONNECTOR
                        the DRM connector
  -D DEVICE, --device DEVICE
                        the DRM device
  --mode MODE           the name of the video mode, e.g., 1920x1080
  --refresh FREQ        the vertical refresh rate in Hz
  -n N, --frames N      run for N frames and exit
  -k UNIFORM, --keyboard UNIFORM
                        add a keyboard
  --touchscreen UNIFORM
                        add a touchscreen
  --trackpad UNIFORM    add a trackpad
  -c UNIFORM FILE, --cubemap UNIFORM FILE
                        add a cubemap
  -t UNIFORM FILE, --texture UNIFORM FILE
                        add a texture
  -v UNIFORM FILE, --volume UNIFORM FILE
                        add a volume
  -m <UNIFORM>.KEY VALUE, --metadata <UNIFORM>.KEY VALUE
                        set uniform metadata
```

</details>

### Examples

```shell
# Basic shader
shadertoy examples/costal_landscape.glsl

# With a texture
shadertoy examples/plasma_globe.glsl -t iChannel0 presets/tex_RGBA_noise_medium.png
```

You can also run it as a Python module:

```shell
python -m shaderbang.shadertoy examples/plasma_globe.glsl
```

Press <kbd>Ctrl</kbd>+<kbd>c</kbd> to exit.
You can explore [shadertoy.com](https://www.shadertoy.com) to find additional shaders.

### Shebangs

Shader files can be made directly executable by adding a shebang line, e.g., in `examples/plasma_globe.glsl`:

```glsl
#!/usr/bin/env -S python -m shaderbang.shadertoy -t iChannel0 presets/tex_RGBA_noise_medium.png
```

Then simply run the shader file:

```shell
./examples/plasma_globe.glsl
```

### Single-file scripts

Python scripts that depend on Shaderbang can embed their dependencies inline using [PEP 723](https://peps.python.org/pep-0723/), so they run without any prior installation.
For example, [`examples/cloth.py`](examples/cloth.py) declares its dependencies in a `# /// script` block:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "shaderbang",
#     "pyopengl",
#     "warp-lang",
# ]
# ///
```

Combined with a shebang, the script becomes directly executable with [uv](https://docs.astral.sh/uv/):

```shell
uv run examples/cloth.py
```

This also works with remote URLs, without cloning the repository:

```shell
uv run https://raw.githubusercontent.com/astefanutti/shaderbang/main/examples/cloth.py
```

### Input devices

Mouse and keyboard devices are automatically detected.
Touchscreen and trackpad devices require explicit uniform mapping via the `--touchscreen` and `--trackpad` options.

Input devices are hot-pluggable: connecting or disconnecting a device at runtime is handled automatically.

## Library

Shaderbang can be used as a library to build custom rendering applications.
The `Input` class is the main extension point: subclass it to define custom rendering logic, input handling, or simulation.

See [`examples/cloth.py`](examples/cloth.py) for a complete example that builds a GPU-accelerated cloth simulation using [NVIDIA Warp](https://github.com/NVIDIA/warp) on top of Shaderbang.

## Compatibility

It's been reported to run successfully on the following configurations:

| Hardware                                    | OS / Kernel                                | Driver                  | Date    |
|---------------------------------------------|--------------------------------------------|-------------------------|---------|
| NVIDIA GeForce RTX 3060                     | Ubuntu 23.10, Linux 6.5                    | NVIDIA Driver 545.29.06 | 03/2024 |
| Jetson Orin NX                              | Jetson Linux 35.3.1, Linux 5.10            | NVIDIA DRM Driver       | 09/2023 |
| Jetson Nano (Tegra X1)                      | Jetson Linux 32.7.4, Linux 4.6             | Mesa NVIDIA Tegra       | 06/2023 |
| Raspberry Pi 4 (Broadcom VideoCore VI)      | Raspberry Pi OS 2023-02, Linux 5.15        | Mesa VC4 V3D 19.3.2     | 09/2023 |
| Raspberry Pi Zero W (Broadcom VideoCore IV) | Raspberry Pi OS 2022-09, Linux 5.15        | Mesa VC4 V3D 19.3.2     | 05/2023 |
| Raspberry Pi 3B+ (Broadcom VideoCore IV)    | Raspberry Pi OS Lite 2020-12, Linux 5.4.79 | Mesa VC4 V3D 19.3.2     | 08/2021 |

## Credits

The DRM/KMS boilerplate code is based on [kmscube](https://gitlab.freedesktop.org/mesa/kmscube/).

The shader examples are copied from the [Shadertoy](https://www.shadertoy.com) website URLs commented at the top of each file.
