import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from typing import Any, Dict, Optional

import torch
# import pytorch_lightning as pl
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from src.data.h5_dataset import h5Dataset, FLUTE_DATASET
from src.data.preprocessor import F0LoudnessRMSPreprocessor


class EncoderDataModule(LightningDataModule):
    """LightningDataModule for URMP dataset.

    A DataModule implements 6 key methods:
        1) def prepare_data(self):
            things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            download data, pre-process, split, save to disk, etc...
        2) def setup(self, stage):
            things to do on every process in DDP
            load data, set variables, etc...
        3) def train_dataloader(self):
            return train dataloader
        4) def val_dataloader(self):
            return validation dataloader
        5) def test_dataloader(self):
            return test dataloader
        6) def teardown(self):
            called on every process in DDP
            clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "dataset/data",
        instrument: str = "flute",
        sample_rate: int = 16000,
        frame_rate: int = 250,
        audio_length: int = 4,
        train_split = 0.80,
        batch_size: int = 16,
        input_keys: tuple = ('audio','loudness','f0','rms'),
        seed = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.sr = sample_rate
        self.train_split = train_split
         
        self.input_keys = input_keys
        self.traindata_path = '{}'.format(f'{data_dir}/train/{instrument}/{sample_rate}sr_{frame_rate}fr_{audio_length}s.h5')
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.seed = seed

    @property
    def get_keys_(self):
        return self.input_keys

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
                dataset = h5Dataset(data_path=self.traindata_path,
                    input_keys=self.input_keys,
                    preprocessor=F0LoudnessRMSPreprocessor(),
                    )

                train_split = int( self.train_split * len(dataset) )
                test_split = ( len(dataset) - train_split ) // 2
                val_split = len(dataset)- train_split - test_split
                self.data_train, self.data_val, self.data_test = random_split(
                    dataset,
                    [train_split,val_split,test_split],
                    generator=torch.Generator().manual_seed(self.seed)
                )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    datamodule = EncoderDataModule(data_dir = FLUTE_DATASET,
                                   train_split = 0.80,
                                   batch_size = 2,
                                   input_keys = ('audio','loudness','f0','rms'),
                                   seed = 42,
                                   num_workers = 0,
                                   pin_memory = False,
    )

    # Call the setup method to prepare the data
    datamodule.setup()

    # Access data loaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    # Test data loaders
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))
    
    # Test cuda
    print("Test cuda")
    print(next(iter(train_loader))['audio'].is_cuda)
    
    # Test cuda
    print("\n","Test cuda")
    print(next(iter(datamodule.train_dataloader()))['audio'].is_cuda)
    
    # Get a batch of data
    for idx, batch in enumerate(train_loader):
        print(idx, batch['audio'].shape)
        print(batch['loudness'].shape)
        print(batch['f0'].shape)
        print(batch['rms'].shape)
        print(batch['audio'].is_cuda)
