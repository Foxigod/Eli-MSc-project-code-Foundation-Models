#!/bin/bash

# batch_sizes=("8" "16" "32")
train_fractions=("1.0" "0.5" "0.25" "0.125" "0.0625")
train_fractions=("0.5")
# train_fractions=("0.5")
# train_fractions=("1.0")
# train_fractions=("0.0625")
seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
seeds=("5" "6" "7")
# seeds=("4" "5" "6" "7")
# seeds=("3")
# seeds=("8")
jobids=()

curr_file="${BASH_SOURCE[0]:-${(%):-%x}}"  # A hybrid bash & zsh approach to get current running file.
curr_dir="$(realpath $(dirname ${curr_file}))"
curr_file_abs="$(realpath ${curr_file})"

log_dir="${curr_dir}/logs_batched"
log_file="${log_dir}/data_efficiency_BaseRec_repeatfailed_v3.log"

# echo ${curr_file}
# echo ${curr_dir_abs}
# echo ${curr_file_abs}

# Check if the file exists and is a regular file
if [ -f "${log_file}" ]; then
  echo "Log file ${log_file} already exists, exiting..."
  exit 1
else
  echo "Log file ${log_file} will be created..."
fi

# for bs in "${batch_sizes[@]}"; do
#     echo "Batch-size: $bs"
#     output=$(sbatch --begin=now+3minutes --job-name=${bs} submit.sh)
#     jobid=${output//[^0-9]/}
#     jobids+=("${jobid}")
# done
lr="6.e-4"
for train_frac in "${train_fractions[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "train_frac:${train_frac}__seed:${seed}"
        # output=$(sbatch --job-name="trfrac_${train_frac}__seed_${seed}" submit.sh ${log_dir} ${train_frac} ${seed})
        output=$(sbatch --job-name="trfrac_${train_frac}__seed_${seed}_BaseRec" ./submit_ViT-B_sen4map_bs80_crop15_patch3_timepatch3_Recreate_baseline.sh \
          --trainer.default_root_dir ${log_dir} \
          --optimizer.init_args.lr ${lr} \
          --data.init_args.train_data_fraction ${train_frac} \
          --data.init_args.val_hdf5_path </DATA_OR_KEY_DIRECTORY/>/val.h5 \
          --data.init_args.val_hdf5_keys_path </DATA_OR_KEY_DIRECTORY/>/val_keys.pkl \
          --seed_everything ${seed})
        jobid=${output//[^0-9]/}
        jobids+=("${jobid}")
        echo "${jobid}:  train_frac:${train_frac}__seed:${seed}" >> "${log_file}"
    done
done

squeue --me

echo ${jobids[*]}
# for jobid in "${jobids[@]}"; do
#     scancel ${jobid}
# done


# echo "TEST"
# echo "This is the output: ${output}"
# echo "The following is just the JobID: ${jobid}"


# canceled=$(scancel ${jobid})
# echo "The submitted job should be canceled already"