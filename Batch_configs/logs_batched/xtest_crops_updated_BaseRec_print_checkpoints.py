from pathlib import Path

# Files that might need changing: 
SEEDS_FILE = Path("./data_efficiency_crops_updated_BaseRec_10seeds.log")

LIGHTNING_LOGS = Path("</logs_batched/lightning_logs/>")


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
    # return epoch_files[sorted(epoch_files.keys())[-1]]
    return sorted(epoch_files.keys())[-1]

with open(SEEDS_FILE, "r") as f:
    chkpts_for_fraction = {}
    for line in f.readlines():
        if line.startswith("#"): continue
        line = line.strip().split()
        jobid = line[0].strip(":")
        line = line[1].split("__")
        train_frac = line[0].split(":")[1]
        seed = line[1].split(":")[1]

        ckpt_no = locate_best_checkpoint_from_jobid(jobid)
        if train_frac not in chkpts_for_fraction: chkpts_for_fraction[train_frac] = []
        chkpts_for_fraction[train_frac].append(ckpt_no)
    for fraction in chkpts_for_fraction:
        print(f"Fraction: {fraction}, checkpoint-epochs: {chkpts_for_fraction[fraction]}")
