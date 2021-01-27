#!/bin/bash
# Break on any error
set -e

DATASET_RESULTS_DIR=/nfs/hpc/share/noelt/data/COSD_Data/data/result_data

# Defining Dataset and split
DATASET=mnist_kl_fixed
SPLIT=2
 
# Creates an archive folder for the dataset
if [ ! -d $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT} ]; then
    mkdir $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}
fi

if [ ! -d $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/checkpoints ]; then
    mkdir $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/checkpoints
fi

if [ ! -d $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/images ]; then
    mkdir $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/images
fi

if [ ! -d $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/evaluations ]; then
    mkdir $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/evaluations
fi

if [ ! -d $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/trajectories ]; then
    mkdir $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/trajectories
fi

#Clear flags
rm -rf ./*_flag

mv ./checkpoints/* $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/checkpoints/
mv ./images/* $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/images/
mv ./evaluations/* $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/evaluations/ 
mv ./generated_images_open_set.dataset $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}
mv ./trajectories/* $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/trajectories/
cp ./params.json $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/params.json
mv ./.last_summ* $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/
mv plot_roc_discriminator* $DATASET_RESULTS_DIR/${DATASET}_split_${SPLIT}/

rm -rf ./checkpoints
rm -rf ./images
rm -rf ./evaluations
rm -rf ./trajectories

echo "Ready for next experiment!"
