set -e
bazel --output_base=/mydata/bazel/output build --config opt --config=numa //tensorflow/tools/lib_package:libtensorflow
sudo tar -C /usr/local -xzf ./bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz