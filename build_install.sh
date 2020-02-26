set -e
bazel --output_user_root=/dev/shm/tfproj/.cache build //tensorflow/tools/pip_package:build_pip_package --verbose_failures
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip uninstall tensorflow -y
pip install /tmp/tensorflow_pkg/tensorflow-2.1.0-cp35-cp35m-linux_x86_64.whl
