from Sen4MapDataset import LucasS2DataModule, Sen4MapDatasetMonthlyComposites
import pickle
import h5py
import numpy as np
import time

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
    
    def get_item_i(i):
        batch = dataset.__getitem__(i)
        label = batch["label"]
        labels[i] = label

    start = time.time()
    for i in range(1000):
        _ = get_item_i(i)
    print(f"Time taken for index 0-1K: {round(time.time()-start, 4)} s")

    start = time.time()
    for i in range(1000, 2000):
        _ = get_item_i(i)
    print(f"Time taken for index 1-2K: {round(time.time()-start, 4)} s")

    start = time.time()
    for i in range(2000, 3000):
        _ = get_item_i(i)
    print(f"Time taken for index 2-3K: {round(time.time()-start, 4)} s")

    start = time.time()
    for i in range(100000,101000):
        _ = get_item_i(i)
    print(f"Time taken for indices 100K-101K: {round(time.time()-start, 4)} s")

    start = time.time()
    for i in range(101000,102000):
        _ = get_item_i(i)
    print(f"Time taken for indices 101K-102K: {round(time.time()-start, 4)} s")

    start = time.time()
    for i in range(200000,201000):
        _ = get_item_i(i)
    print(f"Time taken for indices 200K-201K: {round(time.time()-start, 4)} s")

    start = time.time()
    for i in range(201000,202000):
        _ = get_item_i(i)
    print(f"Time taken for indices 201K-202K: {round(time.time()-start, 4)} s")

    # for i in range(len(dataset)):
    #     if i % 1000 == 0:  print(f"i: {i}")
    #     batch = dataset.__getitem__(i)
    #     label = batch["label"]
    #     labels[i] = label


    