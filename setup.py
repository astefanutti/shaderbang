from distutils.command.build_ext import build_ext as _build_ext
from setuptools import find_packages, setup, Extension, Distribution
from wheel.bdist_wheel import bdist_wheel
import shutil


class CTypesExtension(Extension):
    pass


class ShaderbangBuildExt(_build_ext):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"
        return super().get_ext_filename(ext_name)


class ShaderbangBDistWheel(bdist_wheel):

    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        return "py3", "none", plat

    def run(self):
        super().run()

        # Clean up so we can re-invoke `py -m build --wheel -C--build-option=--platform=...`
        # See https://github.com/pypa/setuptools/issues/1871 for details.
        shutil.rmtree("./build", ignore_errors=True)
        shutil.rmtree("./shaderbang.egg-info", ignore_errors=True)


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True


setup(
    name="shaderbang",
    packages=find_packages(),
    include_package_data=False,
    # package_dir={"shaderbang": "shaderbang"},
    ext_modules=[
        CTypesExtension(
            "shaderbang/_shaderbang",
            [
                "shaderbang/lib/common.c",
                "shaderbang/lib/drm-atomic.c",
                "shaderbang/lib/drm-common.c",
                "shaderbang/lib/drm-legacy.c",
                "shaderbang/lib/shaderbang.c",
            ],
            include_dirs=["/usr/include/libdrm"],
            libraries=[
                "GLESv2",
                "EGL",
                "drm",
                "gbm",
                "pthread",
            ],
        ),
    ],
    distclass=BinaryDistribution,
    cmdclass={
        "build_ext": ShaderbangBuildExt,
        "bdist_wheel": ShaderbangBDistWheel,
    },
    license_files=["LICENSE"],
)
