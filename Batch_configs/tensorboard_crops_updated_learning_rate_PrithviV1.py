from tensorboard_utils import extract_scalar_std_dev, locate_and_open_tfevent, plot_with_errors_to_axis, extract_scalar_iqm_conf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

log_file = Path("./logs_batched/learning_rate_crops_updated_PrithviV1_10seeds.log")
LIGHTNING_LOGS = Path("<./logs_batched/lightning_logs/>")

names = [
    "val/Weighted_Multiclass_F1_Score"
]

names = {
    "val/Weighted_Multiclass_F1_Score": "W.A. F1 Score",
    "val/Classwise_F1_Score_0": "Cereals",
    "val/Classwise_F1_Score_1": "Root Crops",
    "val/Classwise_F1_Score_2": "Nonpermanent Industrial Crops",
    "val/Classwise_F1_Score_3": "Dry Pulses, Vegetables and Flowers",
    "val/Classwise_F1_Score_4": "Fodder Crops",
    "val/Classwise_F1_Score_5": "Bareland",
    "val/Classwise_F1_Score_6": "Woodland and Shrubland",
    "val/Classwise_F1_Score_7": "Grassland",
}

names = {
    "val/Weighted_Multiclass_F1_Score": {"title":"PrithviOS HPO fine-tuning", "y-label":"Val F1 Score"},
    "val/Classwise_F1_Score_0": {"title":"PrithviOS on Cereals", "y-label":"Val F1 Score"},
    "val/Classwise_F1_Score_1": {"title":"PrithviOS on Root Crops", "y-label":"Val F1 Score"},
    "val/Classwise_F1_Score_2": {"title":"PrithviOS on Nonpermanent Industrial Crops", "y-label":"Val F1 Score"},
    "val/Classwise_F1_Score_3": {"title":"PrithviOS on Dry Pulses, Vegetables and Flowers", "y-label":"Val F1 Score"},
    "val/Classwise_F1_Score_4": {"title":"PrithviOS on Fodder Crops", "y-label":"Val F1 Score"},
    "val/Classwise_F1_Score_5": {"title":"PrithviOS on Bareland", "y-label":"Val F1 Score"},
    "val/Classwise_F1_Score_6": {"title":"PrithviOS on Woodland and Shrubland", "y-label":"Val F1 Score"},
    "val/Classwise_F1_Score_7": {"title":"PrithviOS on Grassland", "y-label":"Val F1 Score"},
}

colours = ["#d31f11", "#f47a00", "#62c8d3", "#007191"]  # 4 colours
colours = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]  # 5 colours but yellow too similar to white
colours = ["#b5d1ae", "#80ae9a", "#568b87", "#326677", "#1b485e", "#122740"]  # Teals
colours = ["peru", "darkseagreen", "deepskyblue", "orchid", "lightcoral", "goldenrod"] 

PDFILE = log_file.with_suffix(".pdf")
PDFILE_SCALE = (log_file.parent / (log_file.stem + "_scale")).with_suffix(".pdf")

with open(log_file, "r") as f:
    tfevents:dict[str, list] = {}
    for line in f.readlines():
        if line.startswith("#"): continue
        line = line.strip().split()
        jobid = line[0].strip(":")
        line = line[1].split("__")
        lr = line[0].split(":")[1]
        seed = line[1].split(":")[1]
        if lr not in tfevents:
            tfevents[lr] = []

        version_directory = LIGHTNING_LOGS / f"version_{jobid}"
        tfevent = locate_and_open_tfevent(version_dir=version_directory)
        tfevents[lr].append(tfevent)
    
    
with PdfPages(PDFILE) as pdf:
    with PdfPages(PDFILE_SCALE) as pdf_scale:
        for name in names:
            fig, ax = plt.subplots()
            ax.set_xlabel("Iterations")
            ax.set_ylabel(names[name]["y-label"])
            for i, lr in enumerate(tfevents.keys()):
                # if lr in ["6.e-6", "3.e-5", "6.e-5"]: continue
                print(f"lr: {lr}")
                tfevent_list = tfevents[lr]
                # iterations, mean_scalar_value, std_dev_scalar_value = extract_scalar_std_dev(tfevent_list, name=name)
                iterations, iqmean_scalar_value, confidence_interval_scalar_value = extract_scalar_iqm_conf(tfevent_list, name=name, confidence_level=0.8, bootstrap_method="BCa")
                plot_with_errors_to_axis(ax, iterations, iqmean_scalar_value, confidence_interval_scalar_value, colour=colours[i+1], label=f"LR={lr}", white_region_below=True)
            ax.legend()
            ax.set_title(names[name]["title"])
            ax.grid(True, color="gray", linestyle="dashed", alpha=0.35)
            ax.set_axisbelow(True)
            pdf.savefig(fig)
            ax.set_ylim([0.755, 0.855])
            ax.set_yticks([0.77, 0.79, 0.81, 0.83, 0.85])
            pdf_scale.savefig(fig)