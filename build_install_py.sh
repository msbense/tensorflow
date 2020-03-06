set -e
bazel --output_base=/mydata/bazel/output build --config=opt --config=numa //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /dev/shm/tmp/tensorflow_pkg
pip3 uninstall tensorflow -y
pip3 install /dev/shm/tmp/tensorflow_pkg/tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl
