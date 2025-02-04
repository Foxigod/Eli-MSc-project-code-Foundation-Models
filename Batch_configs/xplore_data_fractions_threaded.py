from Sen4MapDataset import LucasS2DataModule, Sen4MapDatasetMonthlyComposites
import pickle
import h5py
import numpy as np
import concurrent.futures

if __name__ == "__main__":
    with open("</DATA_OR_KEY_DIRECTORY/>/train_keys.pkl", "rb") as f:
        train_keys = pickle.load(f)
    train_file = h5py.File("</DATA_OR_KEY_DIRECTORY/>/train.h5", 'r')

    dataset = Sen4MapDatasetMonthlyComposites(
        h5py_file_object = train_file,
        h5data_keys = train_keys,
    )
    labels = np.ones(len(dataset))*16

    print(len(dataset))
    # exit()

    def io_bound_task(i):
        if i % 1000 == 0: print(f"i: {i}")
        batch = dataset.__getitem__(i)
        label = batch["label"]
        return label

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to be executed concurrently
        results = list(executor.map(io_bound_task, range(len(dataset))))

    labels[:] = results
    # for i in range(len(dataset)):
    #     if i % 1000 == 0:  print(f"i: {i}")
    #     batch = dataset.__getitem__(i)
    #     label = batch["label"]
    #     labels[i] = label

    with open("xplore_results_numpy_threaded.txt", "w") as file:
        file.write(labels)
    