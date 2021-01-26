Updated code from Open Set Learning with Counterfactal Images, ECCV 2018.

start.sh script is set up to run k+1 experiments using the Kliep loss function.

To set up an experiment on split {N} of the {DATASET} dataset, edit the following lines of params.json as follows

"dataset": "{DATA_PATH}/{DATASET}-split{N}a.dataset",
"comparison_dataset": "{DATA_PATH}/{DATASET}-split{N}b.dataset",
"hypothesis": "Example run on {DATASET}",

Also, generativeopenset/training.py is currently padding images to work with mnist. Comment out the two padding lines
