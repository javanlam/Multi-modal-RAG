#!/bin/bash
set -e

git clone https://huggingface.co/lorebianchi98/Talk2DINO-ViTB
git clone https://github.com/facebookresearch/dinov2.git
git clone https://github.com/beichenzbc/Long-CLIP.git

conda create --name multimodal_rag
eval "$(conda shell.bash hook)"
conda activate multimodal_rag

pip install -r requirements.txt

mv Long-CLIP LongCLIP
cd LongCLIP
touch __init__.py

cd checkpoints
git clone https://huggingface.co/BeichenZhang/LongCLIP-B
mv LongCLIP-B/longclip-B.pt .
git clone https://huggingface.co/BeichenZhang/LongCLIP-L
mv LongCLIP-L/longclip-L.pt .