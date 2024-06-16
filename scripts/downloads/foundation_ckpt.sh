#! /bin/bash

# ===============================================================================
# this script downloads the CompletionFormer foundation model checkpoint, trained 
# on NYUDepth-V2 dataset, provided by the original authors, Zhang et al.
# ===============================================================================

mkdir -p ckpts
cd ckpts 
gdown https://drive.google.com/uc?id=1KJUZ4I-v9Nba0DDswHe2-Avq7yll---t
gdown --fuzzy https://drive.google.com/file/d/1raHsLhsI8LUVLShgVyaRS_zciLpJpo2q/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1I1204ezYAmkmDXcMAQpk4EMsxPV3_tSg/view?usp=sharing