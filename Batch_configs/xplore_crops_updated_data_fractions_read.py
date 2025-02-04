import pickle
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

if __name__ == "__main__":
    def extract_per_class_frequencies_for_fractions(pkl_file, fractions):
        with open(pkl_file, "rb") as file:
            labels = pickle.load(file)

        classmap = {
            0: "Cereals",
            1: "Root Crops",
            2: "Non-permanent Industrial Crops",
            3: "Dry Pulses, Vegetables & Flowers",
            4: "Fodder Crops",
            5: "Bareland",
            6: "Woodland & Shrubland",
            7: "Grassland"
        }

        # print(labels)
        print(len(labels))
        # print(np.unique(labels, return_counts=True))

        # fractions = [1.0, 0.5, 0.25, 0.125, 0.0675]
        freqs = []
        classes = classmap.keys()
        for fraction in fractions:
            keys, freq = np.unique(labels[:int(fraction*len(labels))], return_counts=True)
            # print(fraction, freq)
            assert (keys[:] == list(classes)).all(), f"{keys=}, {classes=}"
            freqs.append(freq[:9])
        

        print("-----")
        print(f"{' ':<33} ", end="")
        for fraction in fractions: print(f"{fraction:<8}", end="")
        print()
        for class_no in classmap.keys():
            print(f"{classmap[class_no]:<32}: ", end="")
            for i in range(len(fractions)): print(f"{freqs[i][class_no]:<8}", end="") 
            print()
        print("-----")
        
        # plotting_array = np.array(freqs).T
        # pdf_file = Path("./xplore_results_graph.pdf")
        # with PdfPages(pdf_file) as pdf:
        #     fig, ax = plt.subplots()
        #     for classe, class_evolution in zip(classes, plotting_array):
        #         print(classmap[classe], class_evolution)
        #         ax.scatter(fractions, class_evolution, label=classmap[classe])
        #     ax.legend(bbox_to_anchor=(0.96, 1.09))
        #     # plt.tight_layout()
        #     ax.grid(True)
        #     ax.set_axisbelow(True)
        #     ax.set_yscale("log")
        #     ax.invert_xaxis()
        #     ax.set_xticks(fractions, rotation=-30, ha="left", labels=fractions)
        #     ax.set_ylabel("Samples")
        #     ax.set_xlabel("Train data fraction")
        #     pdf.savefig(fig, bbox_inches="tight")

    extract_per_class_frequencies_for_fractions(
        pkl_file = "./xplore_crops_updated_results_numpy_train.pkl",
        fractions = [1.0, 0.5, 0.25, 0.125, 0.0675]
    )
    extract_per_class_frequencies_for_fractions(
        pkl_file = "./xplore_crops_updated_results_numpy_val.pkl",
        fractions = [1.0, 0.5, 0.25, 0.125, 0.0675]
        # fractions = [1.0]
    )
    extract_per_class_frequencies_for_fractions(
        pkl_file = "./xplore_crops_updated_results_numpy_test.pkl",
        fractions = [1.0, 0.5, 0.25, 0.125, 0.0675]
        # fractions = [1.0]
    )
        