from pathlib import Path
from xtract_utils import interquartile_mean, interquartile_confidence_interval
import numpy as np

SLURM_OUTPUT = Path("../slurm_output")
TEST_LOGS = Path("./test_data_efficiency_PrithviV1_10seeds.log")
# XTRACT_OUTFILE = Path(__file__).parent / (Path(__file__).name.split(".")[0] + "_outfile")
#XTRACT_OUTFILE = Path(__file__).with_suffix(".outfile")
XTRACT_OUTFILE = Path(__file__).with_name(Path(__file__).stem + "_std").with_suffix(".outfile")

interest = {
    "test/Classwise_F1_Score_0":"Artificial land",
    "test/Classwise_F1_Score_1":"Bareland",
    "test/Classwise_F1_Score_2":"Grassland",
    "test/Classwise_F1_Score_3":"Cropland",
    "test/Classwise_F1_Score_4":"Broadleaves",
    "test/Classwise_F1_Score_5":"Conifers",
    "test/Classwise_F1_Score_6":"Shrubland",
    "test/Classwise_F1_Score_7":"Water",
    "test/Classwise_F1_Score_8":"Wetlands",
    "test/Weighted_Multiclass_F1_Score":"W.A. F1 Score",
    "test/Overall_Accuracy":"Overall Accuracy"
}
max_metric_name_length = max(map(len, interest.values()))


file_metrics_per_fraction = {}
with open(TEST_LOGS, "r") as logs:
    for line in logs.readlines():
        if line.startswith("#"): continue
        line = line.strip().split()
        jobid = line[0].strip(":")
        line = line[1].split("__")
        train_frac = line[0].split(":")[1]
        seed = line[1].split(":")[1]
        print(f"Processing jobid: {jobid}, train_frac: {train_frac}, seed: {seed}")
        file_metrics_per_fraction[train_frac] = file_metrics_per_fraction.get(train_frac, {})
        jobout = SLURM_OUTPUT / f"slurm-{jobid}.out"
        file_metrics = {}
        with open(jobout, "r") as outfile:
            found_metrics = False
            for line in outfile.readlines():
                if found_metrics:
                    if "┴" in line: break
                    line = line.strip().split("│")
                    file_metrics[line[1].strip()] = float(line[2].strip())
                    file_metrics_per_fraction[train_frac][line[1].strip()] = file_metrics_per_fraction.get(train_frac).get(line[1].strip(), [])
                    file_metrics_per_fraction[train_frac][line[1].strip()].append(float(line[2].strip()))
                elif "╇" in line:
                    found_metrics = True
'''
with open(XTRACT_OUTFILE, "w") as out:
    for fraction in file_metrics_per_fraction.keys():
        print(f"Fraction: {fraction}", file=out)
        for metric_key, metric_name in interest.items():
            # print(*file_metrics_per_fraction[fraction][metric_key])
            data = np.array(file_metrics_per_fraction[fraction][metric_key])
            conf_low, conf_high = interquartile_confidence_interval(data)
            print(f"{metric_name:<20} = {interquartile_mean(data):.3f},   ({conf_low:.3f}, {conf_high:.3f})", file=out)
        print(file=out)
'''
with open(XTRACT_OUTFILE, "w") as out:
    for fraction in file_metrics_per_fraction.keys():
        print(f"Fraction: {fraction}", file=out)
        for metric_key, metric_name in interest.items():
            # Convert the data to a NumPy array
            data = np.array(file_metrics_per_fraction[fraction][metric_key])
            
            # Calculate the mean and standard deviation
            mean_value = interquartile_mean(data)
            std_value = np.std(data)
            
            # Print the metric name, mean, and standard deviation
            print(f"{metric_name:<32} = {mean_value:.3f},   std = {std_value:.3f}", file=out)
        print(file=out)

# print(file_metrics_per_fraction)
        


# 0: ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# 0: │       test/Average_Accuracy       │        0.43168509006500244        │