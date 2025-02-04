from Sen4MapDataset import LucasS2DataModule, Sen4MapDatasetMonthlyComposites
import pickle
import h5py
import numpy as np
import concurrent.futures

if __name__ == "__main__":
    with open("</key_files/Crop-classification/>/crop_train_keys_updated.pkl", "rb") as f:
        train_keys = pickle.load(f)
    train_file = h5py.File("</DATA_OR_KEY_DIRECTORY/>/train.h5", 'r')

    dataset_train = Sen4MapDatasetMonthlyComposites(
        h5py_file_object = train_file,
        h5data_keys = train_keys,
        classification_map = "crops-updated",
    )
    labels_train = np.ones(len(dataset_train))*16
    print(f"Train dataset size: {len(dataset_train)}")


    with open("</key_files/Crop-classification/>/crop_test_keys_updated.pkl", "rb") as f:
        test_keys = pickle.load(f)
    test_file = h5py.File("</DATA_OR_KEY_DIRECTORY/>/test.h5", 'r')

    dataset_test = Sen4MapDatasetMonthlyComposites(
        h5py_file_object = test_file,
        h5data_keys = test_keys,
        classification_map = "crops-updated",
    )
    labels_test = np.ones(len(dataset_test))*16
    print(f"Test dataset size: {len(dataset_test)}")


    with open("</key_files/Crop-classification/>/crop_val_keys_updated.pkl", "rb") as f:
        val_keys = pickle.load(f)
    val_file = h5py.File("</DATA_OR_KEY_DIRECTORY/>/val.h5", 'r')

    dataset_val = Sen4MapDatasetMonthlyComposites(
        h5py_file_object = val_file,
        h5data_keys = val_keys,
        classification_map = "crops-updated",
    )
    labels_val = np.ones(len(dataset_val))*16
    print(f"Val dataset size: {len(dataset_val)}")
    # exit()

    # # labels[:] = results
    # for i in range(len(dataset)):
    #     if i % 1000 == 0:  print(f"i: {i}")
    #     batch = dataset.__getitem__(i)
    #     label = batch["label"]
    #     labels[i] = label


    # with open("xplore_crops_results_numpy_train.pkl", "wb") as file:
    #     pickle.dump(labels, file)
    