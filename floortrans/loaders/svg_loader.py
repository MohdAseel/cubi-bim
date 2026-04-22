import lmdb
import os
import sys
import pickle
import contextlib
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from numpy import genfromtxt
from floortrans.loaders.house import House


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect OS-level stderr to devnull to silence libpng/OpenCV C-level warnings."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)          # save a copy of fd 2
    os.dup2(devnull_fd, 2)             # point fd 2 → /dev/null
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)      # restore fd 2
        os.close(old_stderr_fd)


class FloorplanSVG(Dataset):
    _LMDB_ENVS = {}

    def __init__(self, data_folder, data_file, is_transform=True,
                 augmentations=None, img_norm=True, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/'):
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.original_size = original_size
        self.format = format
        self.data_folder = data_folder
        self.lmdb_folder = lmdb_folder
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'
        self.lmdb = None

        if format == 'txt':
            self.get_data = self.get_txt
        elif format == 'lmdb':
            self.get_data = self.get_lmdb
            self.is_transform = False

        # Load txt file to list
        self.folders = np.atleast_1d(genfromtxt(data_folder + data_file, dtype='str'))

        if format == 'lmdb':
            self._filter_folders_with_existing_lmdb_keys()

    def __len__(self):
        """__len__"""
        return len(self.folders)

    def __getitem__(self, index):
        sample = self.get_data(index)

        if self.augmentations is not None:
            sample = self.augmentations(sample)
            
        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def get_txt(self, index):
        with _suppress_stderr():
            fplan = cv2.imread(self.data_folder + self.folders[index] + self.image_file_name)
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
        height, width, nchannel = fplan.shape
        fplan = np.moveaxis(fplan, -1, 0)

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)
        # Combining them to one numpy tensor
        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        heatmaps = house.get_heatmap_dict()
        coef_width = 1
        if self.original_size:
            with _suppress_stderr():
                fplan = cv2.imread(self.data_folder + self.folders[index] + self.org_image_file_name)
            fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
            height_org, width_org, nchannel = fplan.shape
            fplan = np.moveaxis(fplan, -1, 0)
            label = label.unsqueeze(0)
            label = torch.nn.functional.interpolate(label,
                                                    size=(height_org, width_org),
                                                    mode='nearest')
            label = label.squeeze(0)

            coef_height = float(height_org) / float(height)
            coef_width = float(width_org) / float(width)
            for key, value in heatmaps.items():
                heatmaps[key] = [(int(round(x*coef_width)), int(round(y*coef_height))) for x, y in value]

        img = torch.tensor(fplan.astype(np.float32))

        sample = {'image': img, 'label': label, 'folder': self.folders[index],
                  'heatmaps': heatmaps, 'scale': coef_width}

        return sample

    def get_lmdb(self, index):
        if self.lmdb is None:
            # Lazy open for multiprocessing pickling issues (esp. on Windows).
            # Reuse one env per absolute path to avoid reopening same env in one process.
            lmdb_path = os.path.abspath(os.path.join(self.data_folder, self.lmdb_folder))
            if lmdb_path not in FloorplanSVG._LMDB_ENVS:
                FloorplanSVG._LMDB_ENVS[lmdb_path] = lmdb.open(
                    lmdb_path, readonly=True,
                    max_readers=8, lock=False,
                    readahead=True, meminit=False
                )
            self.lmdb = FloorplanSVG._LMDB_ENVS[lmdb_path]

        key = self.folders[index].encode()
        with self.lmdb.begin(write=False) as f:
            data = f.get(key)

        sample = pickle.loads(data)
        return sample

    def _filter_folders_with_existing_lmdb_keys(self):
        lmdb_path = os.path.abspath(os.path.join(self.data_folder, self.lmdb_folder))
        if lmdb_path not in FloorplanSVG._LMDB_ENVS:
            FloorplanSVG._LMDB_ENVS[lmdb_path] = lmdb.open(
                lmdb_path, readonly=True,
                max_readers=8, lock=False,
                readahead=True, meminit=False
            )
        self.lmdb = FloorplanSVG._LMDB_ENVS[lmdb_path]

        filtered = []
        with self.lmdb.begin(write=False) as txn:
            for folder in self.folders:
                if txn.get(str(folder).encode()) is not None:
                    filtered.append(folder)

        removed = len(self.folders) - len(filtered)
        if removed > 0:
            print(f"[WARN] Removed {removed} entries missing from LMDB at {lmdb_path}")
        self.folders = np.array(filtered, dtype=str)

    def transform(self, sample):
        fplan = sample['image']
        # Normalization values to range -1 and 1
        fplan = 2 * (fplan / 255.0) - 1

        sample['image'] = fplan

        return sample
