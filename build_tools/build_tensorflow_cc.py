from argparse import ArgumentParser
import os
import subprocess
import shlex


def build_tensorflow_cc(
    bin_path: str, install_prefix: str, cmake_additional_config_args: str
):
    cmake_additional_config_args_list = shlex.split(cmake_additional_config_args)
    print("Configuring tensorflow_cc")
    subprocess.check_call(
        [
            "cmake",
            "-S",
            os.path.join(
                os.path.dirname(__file__), "../third_party/tensorflow_cc/tensorflow_cc"
            ),
            "-B",
            bin_path,
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            "-DTF_URL="
            "https://github.com/tensorflow/tensorflow/archive/v2.12.0-rc0.tar.gz",
            "-DTF_VERSION=2.12.0",
            "-G",
            "Ninja",
        ]
        + cmake_additional_config_args_list
    )
    print("Building tensorflow_cc")
    subprocess.check_call(["cmake", "--build", bin_path])
    print("Installing tensorflow_cc")
    subprocess.check_call(["cmake", "--install", bin_path])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--bin_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../build/third_party/tensorflow_cc"
        ),
    )
    parser.add_argument(
        "--install_prefix",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../build/third_party/tensorflow_cc/install"
        ),
    )
    parser.add_argument(
        "--cmake_additional_config_args",
        type=str,
        default="",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_tensorflow_cc(**vars(args))
