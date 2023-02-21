#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pip install -r "$SCRIPT_DIR/requirements_dev.txt"
pre-commit install
