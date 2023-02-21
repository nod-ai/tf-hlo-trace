from importlib import resources
import tensorflow

tensorflow_package_dir = resources.path(package=tensorflow, resource="").__enter__()
print(tensorflow_package_dir)
