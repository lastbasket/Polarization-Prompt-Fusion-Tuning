#! /bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

DATA_SOURCE_DIR=${1}

if ! [ -e ${DATA_SOURCE_DIR} ] || [ -z $DATA_SOURCE_DIR ]; then
    echo -e ${RED}Invalid data source directory: \(${DATA_SOURCE_DIR}\), cannot create symbolic link${NC}
    exit 1
fi

if [ -d "./data/hammer_polar" ]; then
    echo -e ${GREEN}Symbolic link already exists, that is ./data/hammer_polar, no need to create a new one.${NC}
    exit 0
fi 

mkdir -p data
cd data && ln -s ${DATA_SOURCE_DIR} hammer_polar

echo Done