#!/bin/bash
# Break on any error
set -e

DATASET_DIR=./generativeopenset/data/

# Download any datasets not currently available
# TODO: do this in python, based on --dataset
if [ ! -f $DATASET_DIR/svhn-split0a.dataset ]; then
    python generativeopenset/datasets/download_svhn.py
fi
if [ ! -f $DATASET_DIR/cifar10-split0a.dataset ]; then
     python3 generativeopenset/datasets/download_cifar10.py
fi
if [ ! -f $DATASET_DIR/mnist-split0a.dataset ]; then
    python3 generativeopenset/datasets/download_mnist.py
fi
#if [ ! -f $DATASET_DIR/oxford102.dataset ]; then
#    python generativeopenset/datasets/download_oxford102.py
#fi
#if [ ! -f $DATASET_DIR/celeba.dataset ]; then
#    python generativeopenset/datasets/download_celeba.py
#fi
#if [ ! -f $DATASET_DIR/cifar100-animals.dataset ]; then
#    python generativeopenset/datasets/download_cifar100.py
#fi

# Hyperparameters
GAN_EPOCHS=30
#GAN_EPOCHS=1
CLASSIFIER_EPOCHS=3
CF_COUNT=50
GENERATOR_MODE=open_set
#GENERATOR_MODE=ge_et_al

# Train the intial generative model (E+G+D) and the initial classifier (C_K)
python3 generativeopenset/train_gan.py --epochs $GAN_EPOCHS --result_dir .

mkdir initial_gen_model_done_flag

# Baseline: Evaluate the standard classifier (C_k+1)
python3 generativeopenset/evaluate_classifier.py --result_dir . --mode baseline

mkdir eval_one_done_flag

python3 generativeopenset/evaluate_classifier.py --result_dir . --mode weibull

mkdir eval_two_done_flag

cp checkpoints/classifier_k_epoch_00${GAN_EPOCHS}.pth checkpoints/classifier_kplusone_epoch_00${GAN_EPOCHS}.pth

cp checkpoints/classifier_k_epoch_0030.pth checkpoints/classifier_kplusone_epoch_0030.pth
mkdir checkpoints_copied_flag

## <--- WE ARE HERE
# Generate a number of counterfactual images (in the K+2 by K+2 square grid format)
python3 generativeopenset/generate_${GENERATOR_MODE}.py --result_dir . --count $CF_COUNT

mkdir counterfactual_images_generated_flag

# Automatically label the rightmost column in each grid (ignore the others)
python3 generativeopenset/auto_label.py --output_filename generated_images_${GENERATOR_MODE}.dataset

mkdir column_labeled_flag

# Train a new classifier, now using the aux_dataset containing the counterfactuals
python3 generativeopenset/train_classifier.py --epochs $CLASSIFIER_EPOCHS --aux_dataset generated_images_${GENERATOR_MODE}.dataset

mkdir new_classifier_trained_flag

# Evaluate the C_K+1 classifier, trained with the augmented data
python3 generativeopenset/evaluate_classifier.py --result_dir . --mode kliepmax #fuxin

mkdir classifier_kliepmax_evaluated_flag

python3 generativeopenset/evaluate_classifier.py --result_dir . --mode kliepnorm

mkdir classifier_kliepnorm_evaluated_flag

./print_results.sh

mkdir done_flag
