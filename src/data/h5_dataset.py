from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import torch

FLUTE_DATASET="dataset/data/train/flute/16000sr_250fr_4s.h5"
TRUMPET_DATASET="dataset/data/train/trumpet/16000sr_250fr_4s.h5"
VIOLIN_DATASET="dataset/data/train/violin/16000sr_250fr_4s.h5"

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.preprocessor import F0LoudnessRMSPreprocessor

class h5Dataset(Dataset):
    """ Class to load the h5 dataset pre-computed with dataset/prepare_data.py
    Parameters
    ----------
    sr : int
        audio sample rate
    data_path : str
        path of dataset location
    input_keys : [str]
        list of keys used in data dictionary
    max_audio_val : int
        maximum audio value
    device : str
        load data on cpu or gpu or ddp...
    ---------
    
    """
    def __init__(self, data_path, input_keys, preprocessor, device='cpu'):
        self.data_path = data_path
        self.input_keys = input_keys
        self.device = device
        self.input_data_dicts, self.dataset_len = self.cache_data(self.data_path, len(input_keys))
        self.preprocessor = preprocessor


    def cache_data(self, data_path, nfeatures):
        '''
        Load data to dictionary in RAM
        '''
        h5f = h5py.File(data_path, 'r')
        cache = {}
        keys = h5f.keys()
        nkeys = len(keys)
        ndata = (len(keys)//nfeatures)
        if((nkeys//nfeatures)*nfeatures != nkeys):
            raise Exception("Unexpected dataset len.")

        for key in keys:
            cache[key] = np.array(h5f[key])
        h5f.close()

        return cache, ndata

    def __getitem__(self, idx):
        #Generate current item keys to fetch from RAM cache
        item_keys = [f'{idx}_{k}' for k in self.input_keys ]

        # Load dictionary
        x = {}
        for v,k in enumerate(self.input_keys):
            x[k] = torch.tensor(self.input_data_dicts[item_keys[v]]).unsqueeze(-1).to(self.device)
        return self.preprocessor.run(x)

    def __len__(self):
        return self.dataset_len
    
    
if __name__ == "__main__":
    preprocessor = F0LoudnessRMSPreprocessor()
    dataset = h5Dataset(data_path=FLUTE_DATASET,
                        input_keys=('audio','loudness','f0','rms'),
                        device='cuda:0',
                        preprocessor=preprocessor)
    
    print(f"dataset let: {len(dataset)}")
    print(dataset[0]['f0'].shape)
    print(dataset[0].keys())
    print(dataset[0]['audio'].shape)
    print(dataset[0]['audio'].is_cuda)
    print(type(dataset[0]))
