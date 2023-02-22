from setuptools import setup
from cmake_build_extension import BuildExtension, CMakeExtension
from argparse import ArgumentParser
import sys
import os


def parse_and_comsume_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--build_type",
        type=str,
        choices=["Debug", "Release", "RelWithDebInfo"],
        default="Release",
    )
    args, remaining_args = arg_parser.parse_known_args()
    sys.argv = [sys.argv[-1]] + remaining_args
    return args


args = parse_and_comsume_args()

cmake_config_args = os.environ["TF_HLO_TRACE_CMAKE_CONFIG_ADDITIONAL_ARGS"].split()

setup(
    packages=["tf_hlo_trace"],
    package_dir={"tf_hlo_trace": "src/tf_hlo_trace"},
    install_requires=["tensorflow"],
    requires=[
        "wheel",
        "setuptools",
        "cmake_build_extension",
        "tensorflow",
    ],
    ext_modules=[
        CMakeExtension(
            name="tf_hlo_trace.tf_hlo_trace_py_ext",
            cmake_build_type=args.build_type,
            cmake_configure_options=[
                "-G",
                "Ninja",
            ]
            + cmake_config_args,
        ),
    ],
    cmdclass=dict(build_ext=BuildExtension),
)
