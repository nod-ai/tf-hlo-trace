import tf_hlo_trace  # noqa: F401
import pytest
import os
from jaxlib import xla_extension
import sys


def test_make_source_locations_unique():
    hlo_path = os.path.join(os.path.dirname(__file__), "multi_instuction.hlo")
    with open(hlo_path, "r") as f:
        hlo_str = f.read()
    hlo_module = xla_extension.hlo_module_from_text(hlo_str)
    tf_hlo_trace.make_source_locations_unique(hlo_module)


if __name__ == "__main__":
    pytest.main(sys.argv)
