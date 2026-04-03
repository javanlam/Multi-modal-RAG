#!/bin/bash
set -e

git clone https://huggingface.co/lorebianchi98/Talk2DINO-ViTB
git clone https://github.com/facebookresearch/dinov2.git

conda create --name multimodal_rag
eval "$(conda shell.bash hook)"
conda activate multimodal_rag

cd dinov2
pip install -r requirements.txt

cd ..
pip install -r requirements.txt