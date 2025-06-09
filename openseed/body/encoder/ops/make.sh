#!/usr/bin/env bash
# Build the Deformable DETR CUDA extension for OpenSeeD
# using whatever conda env you’ve activated (vlmaps6 in this case).

if [ -z "$CONDA_PREFIX" ]; then
  echo "ERROR: CONDA_PREFIX is empty. Activate your vlmaps6 env first." >&2
  exit 1
fi

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

NVCC="$(which nvcc)"
if [ -z "$NVCC" ]; then
  echo "ERROR: nvcc not found under \$CONDA_PREFIX/bin. Aborting." >&2
  exit 1
fi
echo "Using nvcc at $NVCC (CUDA_HOME=$CUDA_HOME)"

export PYTHONPATH="$(pwd)"

python setup.py build_ext --inplace

echo "✅ Built MultiScaleDeformableAttention with nvcc @ $NVCC"
