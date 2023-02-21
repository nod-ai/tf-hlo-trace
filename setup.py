import os
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import shutil


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        build_directory = os.path.abspath(self.build_temp)

        cfg = "Debug" if self.debug else "Release"
        cmake_config_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + build_directory,
            "-S",
            f"{os.path.dirname(__file__)}",
            "-B",
            self.build_temp,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=" + cfg,
        ]

        print("-" * 10, "Running CMake prepare", "-" * 40)
        subprocess.check_call(["cmake"] + cmake_config_args)

        print("-" * 10, "Building extensions", "-" * 40)
        build_args = ["--config", cfg]
        self.build_args = build_args
        subprocess.check_call(["cmake", "--build", self.build_temp, "--config", cfg])

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / os.path.basename(self.get_ext_filename(ext.name))
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        shutil.move(source_path, dest_path)


ext_modules = [
    CMakeExtension("tf_hlo_trace.tf_hlo_trace_py_ext"),
]

setup(
    packages=["tf_hlo_trace"],
    package_dir={"tf_hlo_trace": "src/tf_hlo_trace"},
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    install_requires=["tensorflow"],
    zip_safe=False,
)
