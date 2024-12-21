from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
import pyrootutils
import torch
import hydra
import soundfile as sf
import numpy as np
import shutil   
import os
from frechet_audio_distance import FrechetAudioDistance
import csv

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src.data.h5_dataset import h5Dataset, FLUTE_DATASET, TRUMPET_DATASET, VIOLIN_DATASET
from src.data.slice_dataset import SliceDataset
from src.data.preprocessor import F0LoudnessRMSPreprocessor
from src.models.ddsp_decoder import DDSP_Decoder

VERBOSE = True


def remove_keys(mydict):
         return dict((k.removeprefix('model.'), remove_keys(v) if hasattr(v,'keys') else v) for k,v in mydict.items())

def load_model(model_cfg, checkpoint_path, device='cpu') -> DDSP_Decoder:
    """Loads a pre-trained model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint.

    Returns:
        DDSP_Decoder: The model.
    """
    model = hydra.utils.instantiate(model_cfg)
    model_dict = torch.load(checkpoint_path)
    model.load_state_dict(remove_keys(model_dict['state_dict']), strict=False)
    model.eval()
    return model.to(device)


def calculate_fad(target_path: str, resynth_path: str, mode="vggish"):
    """Calculates the Frechet Audio Distance between the target and resynth audio files.

    Args:
        target_path (str): Path to the target audio files.
        resynth_path (str): Path to the resynth audio files.
    """
    # Check if the target directory is empty
    if not os.listdir(target_path):
        print(f"Target directory is empty. No files found.")
        return

    # Check if the resynth directory is empty
    if not os.listdir(resynth_path):
        print(f"Resynth directory is empty. No files found.")
        return

    # Check the mode of the Frechet Audio Distance
    assert mode in ["vggish", "encodec", "pann"], f"Invalid mode: {mode}. Must be either 'vggish' or 'encodec' or 'pann'."
    
    
    # # Check if the background embeddings of the target are available
    # bkg_path = os.path.join(target_path, "bkg.npy")
    # if not os.path.exists(bkg_path):
    #     print(f"[INFO] Saving background embeddings at {bkg_path}")

    if mode == "vggish":
    # Initialize the Frechet Audio Distance object
        frechet = FrechetAudioDistance(
            model_name="vggish",
            # VGGish will resample the files at 16kHz before computing
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        
    elif mode == "encodec":
        # Initialize the Frechet Audio Distance object
        frechet = FrechetAudioDistance(
            model_name="encodec",
            # Audio will be resampled at 24kHz before computing
            sample_rate=24000,
            #use_pca=False,
            #use_activation=False,
            verbose=False,
        )
        
    elif mode == "pann":
        frechet = FrechetAudioDistance(
            model_name="pann",
            sample_rate=16000,
            verbose=False,
        )

    # Calculate the Frechet Audio Distance
    fad_score = frechet.score(target_path, resynth_path)
    # print(f"Frechet Audio Distance: {fad_score}")
    
    return fad_score

def multiple_file_fad(model: DDSP_Decoder, dataloader: torch.utils.data.DataLoader, output_path: str, remove_tree: bool = True, mode="vggish"):
    """Iterates over the dataloader, processes the data with the model, concatenates the output and saves it as an audio file.

    Args:
        model (DDSP_Decoder): The model to use for processing.
        dataloader (torch.utils.data.DataLoader): The dataloader to iterate over.
        output_file (str): The path to the file where the output audio will be saved.
    """
    os.makedirs(output_path, exist_ok=True)
    
    target_dir = os.path.join(output_path, 'target')
    output_dir = os.path.join(output_path, 'output')
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if mode == 'encodec':
        import torchaudio
        resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)

    for batch_idx, batch in enumerate(dataloader):
        if VERBOSE:
            print(f"Processing batch {batch_idx}/ {len(dataloader)}", flush=True)
        output = model(batch)
        
        if mode == 'vggish' or mode == 'pann':
            audio_outputs = np.squeeze(output['synth_audio'].detach().cpu().numpy())
            sf.write(os.path.join(output_dir, f"{batch_idx}.wav"), audio_outputs, 16000)  # Assuming a sample rate of 16000 Hz
            audio_targets = np.squeeze(batch['audio'].detach().cpu().numpy())
            sf.write(os.path.join(target_dir, f"{batch_idx}.wav"), audio_targets, 16000)  # Assuming a sample rate of 16000 Hz
            
        elif mode == 'encodec':
            audio_outputs = resample(output['synth_audio'][0].detach().cpu().T)
            audio_outputs = np.squeeze(audio_outputs.numpy())
            sf.write(os.path.join(output_dir, f"{batch_idx}.wav"), audio_outputs, 24000)  # Assuming a sample rate of 24000 Hz
            audio_targets = resample(batch['audio'][0].detach().cpu().T)
            audio_targets = np.squeeze(audio_targets.numpy())
            sf.write(os.path.join(target_dir, f"{batch_idx}.wav"), audio_targets, 24000)  # Assuming a sample rate of 24000 Hz
       
        
    # calculate fad
    fad = calculate_fad(target_dir, output_dir, mode=mode)
    print(f"FAD_{mode}: {fad}")
      
        
    # remove output/target directory
    if remove_tree:
        shutil.rmtree(target_dir)
        shutil.rmtree(output_dir)
        
    return fad


   
#### Create audio files ############################################################################################################
def create_audio_files(model: DDSP_Decoder, dataloader: torch.utils.data.DataLoader, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    
    target_dir_16 = os.path.join(output_path, 'target_16')
    output_dir_16 = os.path.join(output_path, 'output_16')
    
    target_dir_24 = os.path.join(output_path, 'target_24')
    output_dir_24 = os.path.join(output_path, 'output_24')
    
    os.makedirs(target_dir_16, exist_ok=True)
    os.makedirs(output_dir_16, exist_ok=True)
    os.makedirs(target_dir_24, exist_ok=True)
    os.makedirs(output_dir_24, exist_ok=True)
    
    # Import torchaudio for resampling into 24kHz
    import torchaudio
    resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)

    for batch_idx, batch in enumerate(dataloader):
        if VERBOSE:
            print(f"Processing batch {batch_idx}/ {len(dataloader)}", flush=True)
        output = model(batch)
        
        audio_outputs_16 = np.squeeze(output['synth_audio'].detach().cpu().numpy())
        sf.write(os.path.join(output_dir_16, f"{batch_idx}.wav"), audio_outputs_16, 16000)  # Assuming a sample rate of 16000 Hz
        audio_targets_16 = np.squeeze(batch['audio'].detach().cpu().numpy())
        sf.write(os.path.join(target_dir_16, f"{batch_idx}.wav"), audio_targets_16, 16000)  # Assuming a sample rate of 16000 Hz
            
        audio_outputs_24 = resample(output['synth_audio'][0].detach().cpu().T)
        audio_outputs_24 = np.squeeze(audio_outputs_24.numpy())
        sf.write(os.path.join(output_dir_24, f"{batch_idx}.wav"), audio_outputs_24, 24000)  # Assuming a sample rate of 24000 Hz
        audio_targets_24 = resample(batch['audio'][0].detach().cpu().T)
        audio_targets_24 = np.squeeze(audio_targets_24.numpy())
        sf.write(os.path.join(target_dir_24, f"{batch_idx}.wav"), audio_targets_24, 24000)  # Assuming a sample rate of 24000 Hz
        
    return target_dir_16, output_dir_16, target_dir_24, output_dir_24

### Alternative Main #####################################################################################################################
# def new_main(cfg: DictConfig):
#     # Load the model
#     #model_cfg = OmegaConf.load('/home/greg/PythonProjects/Compression_Algorithms_DDSP/DDSP/MyProjects/Compression_Algorithm_DDSP/configs/model/ddsp/large_hpn.yaml')
#     #checkpoint_path = "/home/greg/PythonProjects/Compression_Algorithms_DDSP/DDSP/MyProjects/Compression_Algorithm_DDSP/logs/large/runs/2024-02-27_17-36-19/checkpoints/epoch_13891.ckpt"
    
#     model_cfg = OmegaConf.load(cfg.model_cfg)
#     model = load_model(model_cfg, cfg.checkpoint_path, device=cfg.device)
#     print("Model loaded. Eval mode: ", not model.training)
    
#     # Load the dataset
#     preprocessor = F0LoudnessRMSPreprocessor()
#     dataset = h5Dataset(data_path=cfg.dataset,
#                         input_keys=('audio','loudness','f0','rms'),
#                         device=cfg.device,
#                         preprocessor=preprocessor) 
    
#     # Create the dataloader
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
#     # Create csv file if it does not exist
#     if not os.path.exists(cfg.output_file):
#         with open(cfg.output_file, 'w') as f:
#             writer = csv.DictWriter(f, fieldnames=["MODEL", "FAD_vggish", "FAD_encodec"])
#             writer.writeheader()
            
#     # Dictionary to store the results
#     results = {}
#     results["MODEL"] = cfg.output_path.split("/")[-1]

#     print(10*"=","Create Audio Files:",10*"=")
#     target_dir_16, output_dir_16, target_dir_24, output_dir_24 = create_audio_files(model, dataloader, output_path=cfg.output_path)
#     print(10*"=","Calculate FAD:",10*"=")
#     if cfg.num_computation > 1:
#         fad_vggish = []
#         fad_encodec = []
#         for i in range(cfg.num_computation):
#             fad_vggish.append(calculate_fad(target_dir_16, output_dir_16, mode="vggish"))
#             fad_encodec.append(calculate_fad(target_dir_24, output_dir_24, mode="encodec"))
        
#         results["FAD_vggish"] = np.mean(fad_vggish)
#         print("FAD_vggish: ", results["FAD_vggish"])
#         results["FAD_encodec"] = np.mean(fad_encodec)
#         print("FAD_encodec: ", results["FAD_encodec"])    
#     else:        
#         results["FAD_vggish"] = calculate_fad(target_dir_16, output_dir_16, mode="vggish")
#         print("FAD_vggish: ", results["FAD_vggish"])
#         results["FAD_encodec"] = calculate_fad(target_dir_24, output_dir_24, mode="encodec")
#         print("FAD_encodec: ", results["FAD_encodec"])
        
#     # Store dictionary in csv file
#     with open(cfg.output_file, 'a') as f:
#         writer = csv.DictWriter(f, fieldnames=["MODEL", "FAD_vggish", "FAD_encodec"])
#         writer.writerow(results)
   
#     if cfg.remove_tree:
#         shutil.rmtree(target_dir_16)
#         shutil.rmtree(output_dir_16)
#         shutil.rmtree(target_dir_24)
#         shutil.rmtree(output_dir_24)
    
 
@hydra.main(version_base="1.3", config_path="./", config_name="fad.yaml")    
def main(cfg: DictConfig):
    # Load the model  
    model_cfg = OmegaConf.load(cfg.model_cfg)
    model = load_model(model_cfg, cfg.checkpoint_path, device=cfg.device)
    if VERBOSE:
        print("Model loaded. Eval mode: ", not model.training)
    
    # Load the dataset
    preprocessor = F0LoudnessRMSPreprocessor()
    dataset = h5Dataset(data_path=cfg.dataset,
                        input_keys=('audio','loudness','f0','rms'),
                        device=cfg.device,
                        preprocessor=preprocessor) 
    
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create csv file if it does not exist
    if not os.path.exists(cfg.output_file):
        with open(cfg.output_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=["MODEL", "FAD_VGGish", "FAD_EnCodec"])
            writer.writeheader()
            
    # Dictionary to store the results
    results = {}
    results["MODEL"] = cfg.output_path.split("/")[-1]

    print(10*"=","COMPUTING FAD(s):",10*"=")
    if cfg.mode == "vggish":
        results["FAD_VGGish"] = multiple_file_fad(model, dataloader, output_path=cfg.output_path, remove_tree=cfg.remove_tree, mode=cfg.mode)
        results["FAD_EnCodec"] = None
        
    elif cfg.mode == "encodec":
        results["FAD_VGGish"] = None
        results["FAD_EnCodec"] = multiple_file_fad(model, dataloader, output_path=cfg.output_path, remove_tree=cfg.remove_tree, mode=cfg.mode)
        
    elif cfg.mode == "both":
        results["FAD_VGGish"] = multiple_file_fad(model, dataloader, output_path=cfg.output_path, remove_tree=cfg.remove_tree, mode="vggish")
        results["FAD_EnCodec"] = multiple_file_fad(model, dataloader, output_path=cfg.output_path, remove_tree=cfg.remove_tree, mode="encodec")
        
    else:
        raise ValueError("Invalid mode. Must be either 'vggish' or 'encodec' or 'both'.")


    # Store dictionary in csv file
    with open(cfg.output_file, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=["MODEL", "FAD_VGGish", "FAD_EnCodec"])
        writer.writerow(results)
    
    
# Hydra config   
if __name__ == "__main__":
    main()


