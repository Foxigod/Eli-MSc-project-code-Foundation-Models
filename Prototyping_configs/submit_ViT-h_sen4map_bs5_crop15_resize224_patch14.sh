#!/bin/bash -x
#SBATCH --job-name=TTsen4map+ViT-h+bs5+crop15+resize224+patch14
#SBATCH --account=<BUDGET>
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24          # <--- I DISLIKE THIS, SINCE IT IS HARDWARE-BASED. I'D LIKE A "MAX" SETTING.
#SBATCH --threads-per-core=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
### ----- ASSERT THAT THE "</slurm_output/>" FOLDER EXISTS -----
#SBATCH -o </slurm_output/>/slurm-%j.out # STDOUT
#SBATCH -e </slurm_output/>/slurm-%j.err # STDERR
### ----- END -----

module --force purge
module load Stages/2024
module load GCC 
module load OpenMPI
module load CUDA

module load PyTorch/2.1.2
source </TerraTorch_ENV/>/bin/activate

module load UCX-settings/RC-CUDA
export UCX_RC_MLX5_FAILURE=INFO
export UCX_RC_MLX5_FC_ENABLE=y
export UCX_RC_MLX5_TIMEOUT=10000000.00us
export UCX_RC_MLX5_RNR_TIMEOUT=10000.00us
export UCX_DC_MLX5_FAILURE=INFO
export UCX_DC_MLX5_FC_ENABLE=y
export UCX_DC_MLX5_TIMEOUT=10000000.00us
export UCX_DC_MLX5_RNR_TIMEOUT=10000.00us
export UCX_UD_MLX5_FAILURE=INFO
export UCX_UD_MLX5_FC_ENABLE=y
export UCX_UD_MLX5_TIMEOUT=10000000.00us
export UCX_UD_MLX5_RNR_TIMEOUT=10000.00us

export NCCL_IB_TIMEOUT=22
export UCX_RC_TIMEOUT=10s
export NCCL_IB_RETRY_CNT=10



# ----- "Without this, srun does not inherit cpus-per-task from sbatch" -----
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export LOGLEVEL=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO

export CUDA_LAUNCH_BLOCKING=1


# ----- Trying to address a communication error -----
MASTER_ADDRi="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDRi="${MASTER_ADDRi}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDRi" | grep -oP '(?<=Address: ).*')"

export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH="<../TerraTorch_additions/>":$PYTHONPATH
srun -l terratorch_patched fit \
        --config ./ViT-h_sen4map_bs5_crop15_resize224_patch14.yaml \
        # --ckpt_path ...