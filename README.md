# Eli Ms project code - Foundation Models

## Repo explanation:
Following is a rundown of the repository, including descriptions of different sections/directories.
### /key_files
Contains key files, describing the train/test/validation split (by containing the IDs used as indices into the .h5 files) for both the entire Sen4Map dataset, which the Land-cover classification task uses, and for a slightly smaller portion of the dataset which is used for the Crop-type classification task. The ...\*`updated`\*.pkl files in the `/Crop-classification/` subdirectory are the most recent/correct version of the crop-type classification task key files. And the .csv files (albeit not named `updated`) are also the same keys, just in a different format.
### /TerraTorch_additions
Contains some additions made to be used with TerraTorch, including a different progress bar class to facilitate real-time monitoring using SLURM's .out file, early version of the dataset and datamodule classes used, and a modification to TerraTorch's classification task to enable logging desired metrics.
### /Prototyping_configs
Contains TerraTorch .yaml configurations used during the Prototyping phase of the validation as well as SLURM .sh scripts to submit said configurations to the compute nodes of JUWELS-BOOSTER HPC at JSC.
### /Batch_configs
Contains further configs and files to submit a collection of experiments. Useful when running with multiple different random seeds and different fractions of the data in the data efficiency experiment. Also contains scripts to read PyTorch-Lightning/Tensorboard logs and turn them into visuals.
#### /Batch_configs/logs_batched
Contains the logs & checkpoints from the fine-tuning runs in a subdirectory (left out of the repo), IDs of runs per experiment (also left out), and the resulting .pdf graphs created from the fine-tuning runs. Also contains scripts to determine the best epoch out of the available ones saved, creating an .sh submission script (for SLURM/sbatch) to run the best epochs/checkpoints on the test dataset to gather results.



## An example of the use of the files from /Batch_configs:
The following script submits jobs to fine-tune the ViT-H model on 5 different learning rates (the simple hyperparameter-optimization), with 5-10 different seeds each (while utilizing a SLURM submissio script and an accompanying TerraTorch .yaml file within the same directory):
`/Batch_configs/batch_crops_updated_learning_rate_ViTh.sh`.
This script also creates the following log file:
`/Batch_configs/logs_batched/learning_rate_crops_updated_ViTh.log`.
This log file contains the list of SLURM job-IDs of the submitted jobs along with the summarized config for each specific ID. This is used for further steps.

The following script reads the .tfevent files of the fine-tuning runs associated with the saved IDs and plots desired metrics into PDFs into the `/Batch_configs/logs_batched/` directory:
`/Batch_configs/tensorboard_crops_updated_learning_rate_ViTh.py`.

Via visual inspection of the performances from those plots, a best learning rate was determined and further used for the data efficiency experiment and reporting results. A similar script was used to submit 4 additional fractions of data on 10 seeds:
`/Batch_configs/batch_crops_updated_data_efficiency_ViTh.sh`
Like before, this script also creates a log file containing a list of SLURM job-IDs of the submitted jobs:
`/Batch_configs/logs_batched/data_efficiency_crops_updated_ViTh.log`

Another script is used to create PDF plots from the .tfevent files associated with the saved IDs:
`/Batch_configs/tensorboard_crops_updated_data_efficiency_ViTh.py`

To report results however, instead of using the procured plots that show the evolution of the performances on the validation data throughout the fine-tuning, we pick the best validation epoch checkpoint and run them in test mode on the test dataset. The following file identifies the best validation epoch for each saved ID, and saves a SLURM submission file that's ready to be submitted:
`/Batch_configs/logs_batched/xtest_crops_updated_ViTh.py`
In this case, that file would get the following name:
`/Batch_configs/logs_batched/xtest_dataeff_cropsUpdated_ViTh.sh`.
This SLURM submission file also saves job-IDs to a log file. These are then used in conjunction with the following file:
`/Batch_configs/logs_batched/xtract_test_crops_updated_ViTh.py`
to create tables reporting the IQM (interquartile mean) (and an 80% bootstrapping confidence interval) of performances on the test-dataset over all seeds and fractions.



> ## Note
> Paths have been modified from original paths into logical placeholders to limit verbosity related to project-specific file-structure. This action has not been thoroughly vetted and may have jumbled up some paths in an incorrect manner.