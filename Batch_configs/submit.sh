#!/bin/bash -x
#SBATCH --job-name=te
#SBATCH --account=<BUDGET>
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24          # <--- I DISLIKE THIS, SINCE IT IS HARDWARE-BASED. I'D LIKE A "MAX" SETTING.
#SBATCH --threads-per-core=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:01:00
### ----- ASSERT THAT THE "./slurm_output/" FOLDER EXISTS -----
#SBATCH -o ./slurm_output/slurm-%j.out # STDOUT
#SBATCH -e ./slurm_output/slurm-%j.err # STDERR
### ----- END -----

srun -l echo "TEST TEST TEST \$1=${1}, \$2=${2}, \${@}=${@}"
# --trainer.default_root_dir ${1}
