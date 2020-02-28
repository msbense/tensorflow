set -e
bazel --output_user_root=/dev/shm/tfnuma/.cache build --config opt //tensorflow/tools/lib_package:libtensorflow
tar -C /dev/shm/libtensorflow -xzf libtensorflow.tar.gz