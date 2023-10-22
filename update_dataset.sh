#!/bin/bash
set -e

mkdir -p data
pushd data
mkdir -p raw

pushd raw
echo "Download the datasets from Alan's account that have been standardized to the rajagopal_no_arms.osim skeleton:"
addb -d dev download --prefix "protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87" "protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/.*\.b3d$"
popd

echo "Post-process the data to low-pass filter at 20 Hz and standardize at 100 Hz:"
addb post-process raw processed --geometry-folder "./Geometry/" --only-dynamics True --sample-rate 100 --recompute-values True --root-history-len 10 --root-history-stride 3
popd

#rm -rf data/train
#rm -rf data/dev
#python3 create_splits.py