#! /bin/bash

# ========================================
# this script downloads the HAMMER dataset
# ========================================
RED='\033[0;31m'
NC='\033[0m'
DATA_FOLDER=${1}

if ! [ -e ${DATA_FOLDER} ] || [ -z "$DATA_FOLDER" ]; then
    echo -e ${RED}The provided data download folder: \(${DATA_FOLDER}\) is invalid, please specify a valid path that exists${NC}
    exit 1
fi

cd $DATA_FOLDER && wget http://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip && unzip _dataset_processed.zip