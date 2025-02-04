from pathlib import Path

# Files that might need changing: 
SEEDS_FILE = Path("./data_efficiency_BaseRec_10seeds.log")
SUBMISSION_FILE = Path("../submit_test_sen4map_bs80_crop15_patch3_timepatch3_Recreate_baseline.sh")
BASH_FILE = Path("./xtest_dataeff_landcover_BaseRec.sh")

LOGS_BATCHED = Path(".")
LIGHTNING_LOGS = Path("</logs_batched/lightning_logs/>")

NEW_LOG_FILE = SEEDS_FILE.parent / f"test_{SEEDS_FILE.name}"

def locate_best_checkpoint_from_jobid(jobid):
    folder = LIGHTNING_LOGS / f"version_{jobid}"
    checkpoints = folder / "checkpoints"
    assert checkpoints.is_dir()
    epoch_files = {}
    ckpts = [path for path in sorted(checkpoints.iterdir()) if path.suffix == ".ckpt"]
    for ckpt in ckpts:
        if ckpt.name.startswith("epoch="): epoch_files[int(ckpt.name.removesuffix(".ckpt").split("=")[-1])] = ckpt
    assert len(epoch_files) > 0
    if len(epoch_files) > 1:
        print(f"epoch_files.keys():  {epoch_files.keys()}")
        print(f"chosen one:  {sorted(epoch_files.keys())[-1]}")
    return epoch_files[sorted(epoch_files.keys())[-1]]

with open(SEEDS_FILE, "r") as f:
    with open(BASH_FILE, "w") as bash:
        bash.write(
f"""
log_file="{NEW_LOG_FILE}"

# Check if the file exists and is a regular file
if [ -f "${{log_file}}" ]; then
  echo "Log file ${{log_file}} already exists, exiting..."
  exit 1
else
  echo "Log file ${{log_file}} will be created..."
fi


"""
        )
        for line in f.readlines():
            if line.startswith("#"): continue
            line = line.strip().split()
            jobid = line[0].strip(":")
            line = line[1].split("__")
            train_frac = line[0].split(":")[1]
            seed = line[1].split(":")[1]

            ckpt_path = locate_best_checkpoint_from_jobid(jobid)
            bash.write(
f"""output=$(sbatch --job-name="test_TerraTorch_seed:{seed}_train_frac:{train_frac}" {SUBMISSION_FILE} \\
    --trainer.default_root_dir {LOGS_BATCHED} \\
    --ckpt_path {ckpt_path} \\
    )
jobid=${{output//[^0-9]/}}
echo "${{jobid}}:  train_frac:{train_frac}__seed:{seed}" >> {NEW_LOG_FILE} \n\n""")


