#!/bin/bash


seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
seeds=("4" "5" "8")
# seeds=("4" "5" "7" "8" "9")
# seeds=("9")
# seeds=("2" "3" "4" "5" "6" "7" "8")
# seeds=("2" "7" "8")
# seeds=("0" "9")
learning_rates=("6.e-6" "3.e-5" "6.e-5" "8.e-5" "6.e-4")
learning_rates=("6.e-4")
# learning_rates=("6.e-4")
# learning_rates=("8.e-5")
# learning_rates=("6.e-5")
# learning_rates=("3.e-5")
# learning_rates=("6.e-6")
jobids=()

curr_file="${BASH_SOURCE[0]:-${(%):-%x}}"  # A hybrid bash & zsh approach to get current running file.
curr_dir="$(realpath $(dirname ${curr_file}))"
curr_file_abs="$(realpath ${curr_file})"

log_dir="${curr_dir}/logs_batched"
log_file="${log_dir}/learning_rate_BaseRec_v2_repeatfailed_v6.log"


# Check if the file exists and is a regular file
if [ -f "${log_file}" ]; then
  echo "Log file ${log_file} already exists, exiting..."
  exit 1
else
  echo "Log file ${log_file} will be created..."
fi


for lr in "${learning_rates[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "lr:${lr}__seed:${seed}"
        # output=$(sbatch --job-name="lr_${lr}__seed_${seed}" ./submit.sh \
        output=$(sbatch --job-name="lr_${lr}__seed_${seed}_BaseRec" ./submit_ViT-B_sen4map_bs80_crop15_patch3_timepatch3_Recreate_baseline.sh \
          --trainer.default_root_dir ${log_dir} \
          --optimizer.init_args.lr ${lr} \
          --seed_everything ${seed})
        jobid=${output//[^0-9]/}
        jobids+=("${jobid}")
        echo "${jobid}:  lr:${lr}__seed:${seed}" >> "${log_file}"
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