from pathlib import Path

files = [
    Path("./xtract_test_land_cover_BaseRec.outfile"),
    Path("./xtract_test_land_cover_PrithviV1.outfile"),
    Path("./xtract_test_land_cover_ViTL-fp32.outfile"),
    Path("./xtract_test_land_cover_ViTh.outfile")
]

OUT_FILE = Path(__file__).with_suffix(".out_combined")

order = [
    "Artificial land",
    "Bareland",
    "Broadleaves",
    "Conifers",
    "Cropland",
    "Grassland",
    "Shrubland",
    "Water",
    "Wetlands",
    "MIDRULE",
    "W.A. F1 Score",
    "MIDRULE",
    "Overall Accuracy",
]

models = {i:{} for i in range(len(files))}
fractions = set()

for i, file in enumerate(files):
    with open(file, "r") as f:
        data = f.read().strip()
    data = data.split("\n\n")
    for frac_results in data:
        frac_results = frac_results.split("\n")
        fraction = frac_results[0].strip().split()[-1]
        fractions.add(fraction)
        frac_results = frac_results[1:]
        models[i][fraction] = {}
        for line in frac_results:
            label, rest = line.split("=")
            label = label.strip()
            iqm, low, high = rest.replace(",", "").replace("(", "").replace(")", "").split()
            models[i][fraction][label] = (iqm, low, high)

with open(OUT_FILE, "w") as out:
    for fraction in sorted(fractions)[::-1]:
        print(f"Fraction: {fraction}", file=out)
        for name in order:
            if name == "MIDRULE":
                print("\midrule", file=out)
                continue
            print(f"{name:<20} ", end="", file=out)
            for i in range(len(models)):
                iqm, low, high = models[i][fraction][name]
                print(f"& \\resultWithConf{'bb' if iqm==sorted([models[j][fraction][name][0] for j in range(len(models))])[-1] else ''}{{{iqm}}}{{{low}}}{{{high}}} ", end="", file=out)
            print("\\\\", file=out)
        print(file=out)
