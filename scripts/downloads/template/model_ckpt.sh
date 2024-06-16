#! /bin/bash

# ===============================================================
# this script downloads the full checkpoint of the proposed model
# ===============================================================

ID=${1}

mkdir -p ckpts
cd ckpts && gdown https://drive.google.com/uc?id=${ID}