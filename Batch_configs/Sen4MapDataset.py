"""
This file was only used to a particular purpose. Creating statistics on the number of samples per class (even for data efficiency experiments),
and is not meant to be used further. (E.g. the batch here only return the label, not the data)
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import h5py
import lightning.pytorch as pl 
from einops import rearrange
from terratorch.datasets.utils import HLSBands
from terratorch.tasks import ClassificationTask
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassFBetaScore, MulticlassJaccardIndex

from torchvision.transforms.v2.functional import resize
from torchvision.transforms.v2 import InterpolationMode

from lightning.pytorch.callbacks import ProgressBar
import pickle


class NewLineProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.test_batch_idx = 0
        self.enabled = True
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.train_batch_idx += 1
        if self.enabled: 
            if self.train_batch_idx==1 or self.train_batch_idx%10==0:
                print(f"Epoch[{trainer.current_epoch}]: Training batch {self.train_batch_idx}/{trainer.num_training_batches}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_batch_idx += 1
        if self.enabled:
            if self.val_batch_idx==1 or self.val_batch_idx%10==0:
                print(f"Epoch[{trainer.current_epoch}]: Validation batch {self.val_batch_idx}/{trainer.num_val_batches}")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_batch_idx += 1
        if self.enabled: 
            if self.test_batch_idx==1 or self.test_batch_idx%10==0:
                print(f"Epoch[{trainer.current_epoch}]: Test batch {self.test_batch_idx}/{trainer.num_test_batches}")

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.test_batch_idx = 0
        if self.enabled: print(f"Epoch {trainer.current_epoch} ended")

    def on_sanity_check_end(self, trainer, pl_module):
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.test_batch_idx = 0
        if self.enabled: print(f"Sanity check ended")

class Sen4MapDatasetMonthlyComposites(Dataset):
    land_use_classification_map={'A10':0, 'A11':0, 'A12':0, 'A13':0, 
    'A20':0, 'A21':0, 'A30':0, 
    'A22':1, 'F10':1, 'F20':1, 
    'F30':1, 'F40':1,
    'E10':2, 'E20':2, 'E30':2, 'B50':2, 'B51':2, 'B52':2,
    'B53':2, 'B54':2, 'B55':2,
    'B10':3, 'B11':3, 'B12':3, 'B13':3, 'B14':3, 'B15':3,
    'B16':3, 'B17':3, 'B18':3, 'B19':3, 'B10':3, 'B20':3, 
    'B21':3, 'B22':3, 'B23':3, 'B30':3, 'B31':3, 'B32':3,
    'B33':3, 'B34':3, 'B35':3, 'B30':3, 'B36':3, 'B37':3,
    'B40':3, 'B41':3, 'B42':3, 'B43':3, 'B44':3, 'B45':3,
    'B70':3, 'B71':3, 'B72':3, 'B73':3, 'B74':3, 'B75':3,
    'B76':3, 'B77':3, 'B80':3, 'B81':3, 'B82':3, 'B83':3,
    'B84':3, 
    'BX1':3, 'BX2':3,
    'C10':4, 'C20':5, 'C21':5, 'C22':5,
    'C23':5, 'C30':5, 'C31':5, 'C32':5,
    'C33':5, 
    'CXX1':5, 'CXX2':5, 'CXX3':5, 'CXX4':5, 'CXX5':5,
    'CXX5':5, 'CXX6':5, 'CXX7':5, 'CXX8':5, 'CXX9':5,
    'CXXA':5, 'CXXB':5, 'CXXC':5, 'CXXD':5, 'CXXE':5,
    'D10':6, 'D20':6, 'D10':6,
    'G10':7, 'G11':7, 'G12':7, 'G20':7, 'G21':7, 'G22':7, 'G30':7, 
    'G40':7,
    'G50':7,
    'H10':8, 'H11':8, 'H12':8, 'H11':8,'H20':8, 'H21':8,
    'H22':8, 'H23':8, '': 9}
    crop_classification_map = {
        "B11":0, "B12":0, "B13":0, "B14":0, "B15":0, "B16":0, "B17":0, "B18":0, "B19":0,  # Cereals
        "B21":1, "B22":1, "B23":1,  # Root Crops
        "B34":2, "B35":2, "B36":2, "B37":2,  # Nonpermanent Industrial Crops
        "B31":3, "B32":3, "B33":3, "B41":3, "B42":3, "B43":3, "B44":3, "B45":3,  # Dry Pulses, Vegetables and Flowers
        "B51":4, "B52":4, "B53":4, "B54":4,  # Fodder Crops
        "F10":5, "F20":5, "F30":5, "F40":5,  # Bareland
        "B71":6, "B72":6, "B73":6, "B74":6, "B75":6, "B76":6, "B77":6, 
        "B81":6, "B82":6, "B83":6, "B84":6, "C10":6, "C20":6, "C30":6, "D10":6, "D20":6,  # Woodland and Shrubland
        "B55":7, "E10":7, "E20":7, "E30":7,  # Grassland
    }
    crop_classification_map_updated = {
        "B11":0, "B12":0, "B13":0, "B14":0, "B15":0, "B16":0, "B17":0, "B18":0, "B19":0,  # Cereals
        "B21":1, "B22":1, "B23":1,  # Root Crops
        "B31":2, "B32":2, "B33":2, "B34":2, "B35":2, "B36":2, "B37":2,  # Nonpermanent Industrial Crops
        "B41":3, "B42":3, "B43":3, "B44":3, "B45":3,  # Dry Pulses, Vegetables and Flowers
        "B51":4, "B52":4, "B53":4, "B54":4,  # Fodder Crops
        "F10":5, "F20":5, "F30":5, "F40":5,  # Bareland
        "B71":6, "B72":6, "B73":6, "B74":6, "B75":6, "B76":6, "B77":6, 
        "B81":6, "B82":6, "B83":6, "B84":6, "C10":6, "C21":6, "C22":6, "C23":6, "C31":6, "C32":6, "C33":6, "D10":6, "D20":6,  # Woodland and Shrubland
        "B55":7, "E10":7, "E20":7, "E30":7,  # Grassland
    }
    # Basically contains three different functions as below:
    def __init__(
            self,
            h5py_file_object:h5py.File,
            h5data_keys = None,
            crop_size:None|int = None,
            dataset_bands:list[HLSBands|int]|None = None,
            input_bands:list[HLSBands|int]|None = None,
            resize = False,
            resize_to = [224, 224],
            resize_interpolation = InterpolationMode.BILINEAR,
            resize_antialiasing = True,
            reverse_tile = False,
            reverse_tile_size = 3,
            classification_map = "land-use"
            ):
        # print(f"Sen4MapDatasetMonthlyComposites.__init__() was just called...")
        # Here data loading happens
        self.h5data = h5py_file_object
        # print(f"Sen4MapDatasetMonthlyComposites.__init__() progression no. 1...")
        if h5data_keys is None:
            # print(f"Sen4MapDatasetMonthlyComposites.__init__() progression no. 1.2 ...")
            self.h5data_keys = list(self.h5data.keys())
        else:
            self.h5data_keys = h5data_keys
        # print(f"Sen4MapDatasetMonthlyComposites.__init__() progression no. 2...")
        self.crop_size = crop_size
        # print(f"Sen4MapDatasetMonthlyComposites.__init__() progression no. 3...")
        if input_bands and not dataset_bands:
            raise ValueError(f"input_bands was provided without specifying the dataset_bands")
        # print(f"Sen4MapDatasetMonthlyComposites.__init__() progression no. 4...")
        # self.dataset_bands = dataset_bands
        # self.input_bands = input_bands
        if input_bands and dataset_bands:
            # print(f"Sen4MapDatasetMonthlyComposites.__init__() progression no. 5...")
            self.input_channels = [dataset_bands.index(band_ind) for band_ind in input_bands if band_ind in dataset_bands]
        else: self.input_channels = None

        classification_maps = {"land-use": Sen4MapDatasetMonthlyComposites.land_use_classification_map,
                               "crops": Sen4MapDatasetMonthlyComposites.crop_classification_map,
                               "crops-updated": Sen4MapDatasetMonthlyComposites.crop_classification_map_updated}
        if classification_map not in classification_maps.keys():
            raise ValueError(f"Provided classification_map of: {classification_map}, is not from the list of valid ones: {classification_maps.keys()}")
        self.classification_map = classification_maps[classification_map]

        self.resize = resize
        self.resize_to = resize_to
        self.resize_interpolation = resize_interpolation
        self.resize_antialiasing = resize_antialiasing
        
        self.reverse_tile = reverse_tile
        self.reverse_tile_size = reverse_tile_size
        # print(f"Sen4MapDatasetMonthlyComposites.__init__() just finished...")

    def __getitem__(self, index):
        # we can call dataset with an index, eg. dataset[0]
        # print(f"Sen4MapDatasetMonthlyComposites.__getitem__() was just called...")
        # print(f"index: {index},  type(index): {type(index)}")
        # print(f"self.h5data_keys[index]: {self.h5data_keys[index]},  type(self.h5data_keys[index]): {type(self.h5data_keys[index])}")
        im = self.h5data[self.h5data_keys[index]]
        Label = self.get_data(im)
        Label = torch.LongTensor(Label)

        return {"label":Label}

    def __len__(self):
        # print(f"Sen4MapDatasetMonthlyComposites.__len__() was just called...")
        # Here we can call len(dataset)
        return len(self.h5data_keys)

    def get_data(self, im):
        Label = im.attrs['lc1']
        Label = self.classification_map[Label]
        Label = np.array(Label)
        Label = Label.astype('float32')

        return Label
    
    def crop_center(self, img_b:torch.Tensor, cropx, cropy) -> torch.Tensor:
        c, t, y, x = img_b.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img_b[0:c, 0:t, starty:starty+cropy, startx:startx+cropx]
    
    # def reverse_tiling_pytorch(self, img: torch.Tensor, kernel_size: int=3):
    #     assert kernel_size % 2 == 1
    #     assert kernel_size >= 3
    #     padding = (kernel_size - 1) // 2
    #     B, C, H, W = img.shape
    #     # img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    #     img = F.pad(img, pad=(padding,padding,padding,padding), mode="replicate")  # Replicate pad with border-width 1
    #     # Apply unfold to get kernel_size x kernel_size patches
    #     unfold = F.unfold(img, kernel_size=kernel_size, padding=0)  # 1x1x(H*W) output
    #     patches = unfold.permute(0, 2, 1)  # Shape (1, H*W, kernel_size^2)
    #     # Reshape and rearrange patches into H*kernel_size x W*kernel_size
    #     patches = patches.view(B*C, H, W, kernel_size, kernel_size)
    #     expanded_img = patches.permute(0, 1, 3, 2, 4).reshape(B, C, H*kernel_size, W*kernel_size)
    #     return expanded_img.squeeze()
    
    def reverse_tiling_pytorch(self, img_tensor: torch.Tensor, kernel_size: int=3):
        assert kernel_size % 2 == 1
        assert kernel_size >= 3
        padding = (kernel_size - 1) // 2
        # img_tensor shape: (batch_size, channels, H, W)
        batch_size, channels, H, W = img_tensor.shape
        # Unfold: Extract 3x3 patches with padding of 1 to cover borders
        img_tensor = F.pad(img_tensor, pad=(padding,padding,padding,padding), mode="replicate")
        patches = F.unfold(img_tensor, kernel_size=kernel_size, padding=0)  # Shape: (batch_size, channels*9, H*W)
        # Reshape to organize the 9 values from each 3x3 neighborhood
        patches = patches.view(batch_size, channels, kernel_size*kernel_size, H, W)  # Shape: (batch_size, channels, 9, H, W)
        # Rearrange the patches into (batch_size, channels, 3, 3, H, W)
        patches = patches.view(batch_size, channels, kernel_size, kernel_size, H, W)
        # Permute to have the spatial dimensions first and unfold them
        patches = patches.permute(0, 1, 4, 2, 5, 3)  # Shape: (batch_size, channels, H, 3, W, 3)
        # Reshape to get the final expanded image of shape (batch_size, channels, H*3, W*3)
        expanded_img = patches.reshape(batch_size, channels, H * kernel_size, W * kernel_size)
        return expanded_img

    def min_max_normalize(self, tensor:torch.Tensor, q_low:list[float], q_hi:list[float]) -> torch.Tensor:
        dtype = tensor.dtype
        q_low = torch.as_tensor(q_low, dtype=dtype, device=tensor.device)
        q_hi = torch.as_tensor(q_hi, dtype=dtype, device=tensor.device)
        x = torch.tensor(-12.0)
        y = torch.exp(x)
        tensor.sub_(q_low[:, None, None, None]).div_((q_hi[:, None, None, None].sub_(q_low[:, None, None, None])).add(y))
        return tensor



class LucasS2DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            batch_size,
            num_workers,
            prefetch_factor = 0,
            # dataset_bands:list[HLSBands|int] = None,
            # input_bands:list[HLSBands|int] = None,
            train_hdf5_path = None,
            train_hdf5_keys_path = None,
            test_hdf5_path = None,
            test_hdf5_keys_path = None,
            val_hdf5_path = None,
            val_hdf5_keys_path = None,
            **kwargs
            ):
        # print(f"LucasS2DataModule.__init__() was just called...")
        super(LucasS2DataModule).__init__()
        # print(f"super(LucasS2DataModule).__init__() just finished...")
        self.prepare_data_per_node = False
        self._log_hyperparams = None
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.train_hdf5_path = train_hdf5_path
        self.test_hdf5_path = test_hdf5_path
        self.val_hdf5_path = val_hdf5_path

        self.train_hdf5_keys_path = train_hdf5_keys_path
        self.test_hdf5_keys_path = test_hdf5_keys_path
        self.val_hdf5_keys_path = val_hdf5_keys_path

        if train_hdf5_path and not train_hdf5_keys_path: raise ValueError(f"Train dataset path provided but not the path to the dataset keys")
        if test_hdf5_path and not test_hdf5_keys_path: raise ValueError(f"Test dataset path provided but not the path to the dataset keys")
        if val_hdf5_path and not val_hdf5_keys_path: raise ValueError(f"Val dataset path provided but not the path to the dataset keys")

        self.shuffle = kwargs.pop("shuffle", None)
        self.train_shuffle = kwargs.pop("train_shuffle", None) or self.shuffle
        self.val_shuffle = kwargs.pop("val_shuffle", None)
        self.test_shuffle = kwargs.pop("test_shuffle", None)

        self.train_data_fraction = kwargs.pop("train_data_fraction", 1.0)
        self.val_data_fraction = kwargs.pop("val_data_fraction", 1.0)
        self.test_data_fraction = kwargs.pop("test_data_fraction", 1.0)


        all_hdf5_data_path = kwargs.pop("all_hdf5_data_path", None)
        if all_hdf5_data_path is not None:
            print(f"all_hdf5_data_path provided, will be interpreted as the general data path for all splits.\nKeys in provided train_hdf5_keys_path assumed to encompass all keys for entire data. Validation and Test keys will be subtracted from Train keys.")
            if self.train_hdf5_path: raise ValueError(f"Both general all_hdf5_data_path provided and a specific train_hdf5_path, remove the train_hdf5_path")
            if self.val_hdf5_path: raise ValueError(f"Both general all_hdf5_data_path provided and a specific val_hdf5_path, remove the val_hdf5_path")
            if self.test_hdf5_path: raise ValueError(f"Both general all_hdf5_data_path provided and a specific test_hdf5_path, remove the test_hdf5_path")
            self.train_hdf5_path = all_hdf5_data_path
            self.val_hdf5_path = all_hdf5_data_path
            self.test_hdf5_path = all_hdf5_data_path
            self.reduce_train_keys = True
        else:
            self.reduce_train_keys = False

        self.resize = kwargs.pop("resize", False)
        self.resize_to = kwargs.pop("resize_to", None)
        if self.resize and self.resize_to is None:
            raise ValueError(f"Config provided resize as True, but resize_to parameter not given")
        self.resize_interpolation = kwargs.pop("resize_interpolation", None)
        if self.resize and self.resize_interpolation is None:
            print(f"Config provided resize as True, but resize_interpolation mode not given. Will assume default bilinear")
            self.resize_interpolation = "bilinear"
        interpolation_dict = {
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "nearest": InterpolationMode.NEAREST,
            "nearest_exact": InterpolationMode.NEAREST_EXACT
        }
        if self.resize:
            if self.resize_interpolation not in interpolation_dict.keys():
                raise ValueError(f"resize_interpolation provided as {self.resize_interpolation}, but valid options are: {interpolation_dict.keys()}")
            self.resize_interpolation = interpolation_dict[self.resize_interpolation]
        self.resize_antialiasing = kwargs.pop("resize_antialiasing", True)


        self.kwargs = kwargs

    def _load_hdf5_keys_from_path(self, path, fraction=1.0):
        with open(path, "rb") as f:
            keys = pickle.load(f)
            return keys[:int(fraction*len(keys))]

    def setup(self, stage: str):
        # print(f"LucasS2DataModule.setup(stage) was just called with stage={stage} ...")
        if stage == "fit":
            # print(f"stage == 'fit'...")
            train_keys = self._load_hdf5_keys_from_path(self.train_hdf5_keys_path, fraction=self.train_data_fraction)
            # val_keys = self._load_hdf5_keys_from_path(self.val_hdf5_keys_path, fraction=self.val_data_fraction)
            # if self.reduce_train_keys:
            #     test_keys = self._load_hdf5_keys_from_path(self.test_hdf5_keys_path, fraction=self.test_data_fraction)
            #     train_keys = list(set(train_keys) - set(val_keys) - set(test_keys))
            train_file = h5py.File(self.train_hdf5_path, 'r')
            self.lucasS2_train = Sen4MapDatasetMonthlyComposites(
                train_file, 
                h5data_keys = train_keys, 
                resize = self.resize,
                resize_to = self.resize_to,
                resize_interpolation = self.resize_interpolation,
                resize_antialiasing = self.resize_antialiasing,
                **self.kwargs
            )
            # val_file = h5py.File(self.val_hdf5_path, 'r')
            # self.lucasS2_val = Sen4MapDatasetMonthlyComposites(
            #     val_file, 
            #     h5data_keys=val_keys, 
            #     resize = self.resize,
            #     resize_to = self.resize_to,
            #     resize_interpolation = self.resize_interpolation,
            #     resize_antialiasing = self.resize_antialiasing,
            #     **self.kwargs
            # )
        if stage == "test":
            test_file = h5py.File(self.test_hdf5_path, 'r')
            test_keys = self._load_hdf5_keys_from_path(self.test_hdf5_keys_path, fraction=self.test_data_fraction)
            self.lucasS2_test = Sen4MapDatasetMonthlyComposites(
                test_file, 
                h5data_keys=test_keys, 
                resize = self.resize,
                resize_to = self.resize_to,
                resize_interpolation = self.resize_interpolation,
                resize_antialiasing = self.resize_antialiasing,
                **self.kwargs
            )

    def train_dataloader(self):
        return DataLoader(self.lucasS2_train, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, shuffle=self.train_shuffle)

    def val_dataloader(self):
        return DataLoader(self.lucasS2_val, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, shuffle=self.val_shuffle)

    def test_dataloader(self):
        return DataLoader(self.lucasS2_test, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, shuffle=self.test_shuffle)
    



class CustomClassificationTask(ClassificationTask):
    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {
                "Overall_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "Average_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Multiclass_Accuracy_Class": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                ),
                "Multiclass_Jaccard_Index": MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index),
                "Multiclass_Jaccard_Index_Class": ClasswiseWrapper(
                    MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                ),
                # why FBetaScore
                "Multiclass_F1_Score": MulticlassFBetaScore(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    beta=1.0,
                    average="micro",
                ),
                "Weighted_Multiclass_F1_Score": MulticlassFBetaScore(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    beta=1.0,
                    average="weighted",
                ),
                "Multiclass_F1_Score_Class": ClasswiseWrapper(
                    MulticlassFBetaScore(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        beta=1.0,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Classwise_F1_Score_",
                )
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")



