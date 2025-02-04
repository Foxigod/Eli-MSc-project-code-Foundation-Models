import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

if __name__ == "__main__":
    with open("xplore_results_numpy.pkl", "rb") as file:
        labels = pickle.load(file)

    classmap = {
        0: "Artificial land",
        1: "Bareland",
        4: "Broadleaves",
        5: "Conifers",
        3: "Cropland",
        2: "Grassland",
        6: "Shrubland",
        7: "Water",
        8: "Wetlands"
    }

    print(labels)
    print(len(labels))
    print(np.unique(labels, return_counts=True))

    fractions = [1.0, 0.5, 0.25, 0.125, 0.0675]
    freqs = []
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for fraction in fractions:
        keys, freq = np.unique(labels[:int(fraction*len(labels))], return_counts=True)
        print(fraction, freq)
        assert (keys[:9] == classes).all()
        freqs.append(freq[:9])
    exit()
    
    plotting_array = np.array(freqs).T
    pdf_file = Path("./xplore_results_graph.pdf")
    with PdfPages(pdf_file) as pdf:
        fig, ax = plt.subplots()
        for classe, class_evolution in zip(classes, plotting_array):
            print(classmap[classe], class_evolution)
            ax.scatter(fractions, class_evolution, label=classmap[classe])
        ax.legend(bbox_to_anchor=(0.96, 1.09))
        # plt.tight_layout()
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks(fractions, rotation=-30, ha="left", labels=fractions)
        ax.set_ylabel("Samples")
        ax.set_xlabel("Train data fraction")
        pdf.savefig(fig, bbox_inches="tight")
        