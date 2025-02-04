from Sen4MapDataset import LucasS2DataModule, Sen4MapDatasetMonthlyComposites
import pickle
import h5py
import numpy as np
import concurrent.futures

if __name__ == "__main__":
    def extract_amounts(keys_pkl, hdf5file, outfile_pkl):
        with open(keys_pkl, "rb") as f:
            train_keys = pickle.load(f)
        train_file = h5py.File(hdf5file, 'r')

        dataset = Sen4MapDatasetMonthlyComposites(
            h5py_file_object = train_file,
            h5data_keys = train_keys,
            classification_map = "crops-updated",
        )
        labels = np.ones(len(dataset))*16

        print(len(dataset))

        # labels[:] = results
        for i in range(len(dataset)):
            if i % 1000 == 0:  print(f"i: {i}")
            batch = dataset.__getitem__(i)
            label = batch["label"]
            labels[i] = label

        with open(outfile_pkl, "wb") as file:
            pickle.dump(labels, file)

    extract_amounts(
        keys_pkl = "</key_files/Crop-classification/>/crop_train_keys_updated.pkl",
        hdf5file = "</DATA_OR_KEY_DIRECTORY/>/train.h5",
        outfile_pkl = "xplore_crops_updated_results_numpy_train.pkl"
    )
    extract_amounts(
        keys_pkl = "</key_files/Crop-classification/>/crop_test_keys_updated.pkl",
        hdf5file = "</DATA_OR_KEY_DIRECTORY/>/test.h5",
        outfile_pkl = "xplore_crops_updated_results_numpy_test.pkl"
    )
    extract_amounts(
        keys_pkl = "</key_files/Crop-classification/>/crop_val_keys_updated.pkl",
        hdf5file = "</DATA_OR_KEY_DIRECTORY/>/val.h5",
        outfile_pkl = "xplore_crops_updated_results_numpy_val.pkl"
    )
    