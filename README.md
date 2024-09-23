# KMS GLSL

KMS GLSL is a command line tool that runs OpenGL fragment shaders, using the [DRM/KMS Linux kernel subsystem](https://en.wikipedia.org/wiki/Direct_Rendering_Manager).
It runs shaders fullscreen, and does not require any windowing system, like X or Wayland.

It has initially been developed to run shaders from [Shadertoy](https://www.shadertoy.com), on the [Raspberry Pi](https://ttt.io/glsl-raspberry-pi).
However, it works with any GPU and display controller hardware, provided a DRM/KMS driver is available, like on the [Jetson Nano](https://ttt.io/glsl-jetson-nano).
Examples of configuration, where it's been reported to run successfully on, are listed in the [compatibility](#compatibility) section.

In the following picture, this [Shadertoy shader](https://www.shadertoy.com/view/MsX3Wj) runs on the Raspberry Pi 4, connected to the official Raspberry Pi 7" touchscreen monitor, in WVGA resolution:

![A Shadertoy shader running on the Raspberry Pi 4](./raspberry_pi.jpg)

[Another shader](https://www.shadertoy.com/view/fstyD4) that runs on the Jetson Nano in full HD resolution:

![A Shadertoy shader running on the Jetson Nano](./jetson_nano.jpg)

## Build

You need to clone the project, and run the following commands:

```shell
$ sudo apt update
# Install the build tools
$ sudo apt install gcc make
# Install the required DRM, GBM, EGL and OpenGL ES API headers
$ sudo apt install libdrm-dev libgbm-dev libegl-dev libgles2-mesa-dev
# Install the X C binding and RandR extension header / library files (optional)
$ sudo apt install libxcb-randr0-dev
# Build the glsl binary and library
$ make
```

## Usage

Once you've successfully built the binary / library, you can either run it directly, or use the Python wrapper, that adds a layer for managing shader inputs, that you can also extend to add your own custom inputs.

### Native

```console
$ ./glsl -h
Usage: ./glsl [-aACDfmnpvx] <shader_file>

options:
    -a, --async              use async page flipping
    -A, --atomic             use atomic mode setting and fencing
    -C, --connector=ID       use the connector with the provided ID (see drm_info)
    -D, --device=DEVICE      use the given device
    -f, --format=FOURCC      framebuffer format
    -h, --help               print usage
    -m, --modifier=MODIFIER  hardcode the selected modifier
    -n, --frames=N           run for the given number of frames and exit
    -p, --perfcntr=LIST      sample specified performance counters using
                             the AMD_performance_monitor extension (comma
                             separated list)
    -v, --vmode=VMODE        specify the video mode in the format
                             <mode>[-<vrefresh>]
    -x, --surfaceless        use surfaceless mode, instead of GBM surface
```

> [!NOTE]
> [Shaders](https://www.shadertoy.com/howto#q1) from [Shadertoy](https://www.shadertoy.com/) are currently expected as input shader files.

You can try it with the shaders available in the `examples` directory, e.g.:

```shell
$ ./glsl examples/costal_landscape.glsl
```

Press <kbd>Ctrl</kbd>+<kbd>c</kbd> to exit the program.
You can explore [shadertoy.com](https://www.shadertoy.com) to find additional shaders.
Note the shaders from the `examples` directory assume OpenGL ES 3.1 support, and may not work with lower versions of the specification.

No inputs can be provided using the native CLI directly.
You can use the Python wrapper, that adds a layer around the native library for managing shader inputs, as explained below.

### Python

```console
$ python glsl.py -h
usage: glsl.py [-h] [--async-page-flip | --no-async-page-flip]
               [--atomic-drm-mode | --no-atomic-drm-mode] [-C CONNECTOR]
               [-D DEVICE] [--mode MODE] [-n N] [-k UNIFORM]
               [--touchscreen UNIFORM] [--trackpad UNIFORM] [-c UNIFORM FILE]
               [-t UNIFORM FILE] [-v UNIFORM FILE] [-m <UNIFORM>.KEY VALUE]
               FILE

Run OpenGL shaders using DRM/KMS

positional arguments:
  FILE                  the shader file

options:
  -h, --help            show this help message and exit
  --async-page-flip, --no-async-page-flip
                        use async page flipping
  --atomic-drm-mode, --no-atomic-drm-mode
                        use atomic mode setting and fencing
  -C CONNECTOR, --connector CONNECTOR
                        the DRM connector
  -D DEVICE, --device DEVICE
                        the DRM device
  --mode MODE           specify the video mode in the format
                        <resolution>[-<vrefresh>]
  -n N, --frames N      run for the given number of frames and exit
  -k UNIFORM, --keyboard UNIFORM
                        add keyboard
  --touchscreen UNIFORM
                        add touchscreen device
  --trackpad UNIFORM    add trackpad device
  -c UNIFORM FILE, --cubemap UNIFORM FILE
                        add cubemap
  -t UNIFORM FILE, --texture UNIFORM FILE
                        add texture
  -v UNIFORM FILE, --volume UNIFORM FILE
                        add volume
  -m <UNIFORM>.KEY VALUE, --metadata <UNIFORM>.KEY VALUE
                        set uniform metadata
```

> [!NOTE]
> Python 3.10+ is required.

You'll have to install the required dependencies once, e.g., with [venv](https://docs.python.org/3.10/library/venv.html):

```shell
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install libevdev pillow
```

You may also want to be in the `input` group, so `/dev/input/eventX` devices can be open to handle [evdev](https://docs.kernel.org/input/input.html#evdev) events, without running as root, e.g.:

```shell
$ sudo adduser $USER input
```

You can then try it with the shaders available in the `examples` directory, e.g.:

```shell
$ python shadertoy.py examples/plasma_globe.glsl -t iChannel0 presets/tex_RGBA_noise_medium.png
```

Press <kbd>Ctrl</kbd>+<kbd>c</kbd> to exit the program.
You can explore [shadertoy.com](https://www.shadertoy.com) to find additional shaders.

If you want to add your own inputs, you can find the documentation and some examples in the `glsl.py` file.

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

## Roadmap

- Add support for texture buffers
- Add support for audio / video inputs
- Parse GLSL files to retrieve uniforms metadata

## Credits

The DRM/KMS ceremony code is copied from [kmscube](https://gitlab.freedesktop.org/mesa/kmscube/).

The shader examples are copied from the [Shadertoy](https://www.shadertoy.com) website URLs commented at the top of each file.
