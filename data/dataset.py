import sys
import warnings

from torch.utils.data import DataLoader

sys.path.append('.') # 

from data.build_h5py_data import H5PYMixedSliceData

warnings.filterwarnings("ignore")

def get_h5py_mixed_dataset(dataset_name):
    train_dataset=H5PYMixedSliceData(dataset_name)
    val_loader = DataLoader(H5PYMixedSliceData(dataset_name, validation=True), batch_size=1, shuffle=False)
    return train_dataset, val_loader

def get_h5py_mixed_test_dataset(dataset_name):
    val_loader = DataLoader(H5PYMixedSliceData(dataset_name, test=True), batch_size=1, shuffle=False)
    return val_loader
