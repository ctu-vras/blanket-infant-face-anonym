#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate facefusion
module load FFmpeg/5.1.2-GCCcore-11.3.0-CUDA-11.7.0
export GRADIO_TEMP_DIR=./tmp
python "$@"
