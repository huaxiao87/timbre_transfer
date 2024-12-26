______________________________________________________________________

<div align="center">

# Distilling DDSP: Exploring Real-Time Audio Generation on Embedded Systems
#### Gregorio Andrea Giudici - Franco Caspe - Leonardo Gabrielli - Stefano Squartini - Luca Turchet
<!--<span style="font-size: 20px;"><b>[>>website<<](https://gregogiudici.github.io/distilling-ddsp)</b></span><br>-->
<!--<i>(with audio samples and more)</i>-->
### 🎵🎧 **[>>Website<<](https://gregogiudici.github.io/distilling-ddsp)** 🎧🎵  
<i>(with audio samples and more)</i>


<center>
<img src="docs\misc\images\distillation_scheme.png"">
</center>
<br>

</div>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<!-- <a href="https://magenta.tensorflow.org/ddsp"><img alt="DDSP" src="https://img.shields.io/badge/DDSP-Magenta-792ee5"></a> -->

## Description
#### Repository Structure:

 * [configs](configs/): Contains configuration files for the project. This projects uses. This project uses [hydra](https://hydra.cc/) to personalize dataset generation, and build and train models. It is reccomended to take a look at the available options in yaml files before processing a dataset or training a model.
 * [dataset](dataset/): Includes code for generating and preprocessing datasets.
 * [docs](docs/): Documentation for the project's webpage.
 * [metrics](metrics/): Contains metrics used to analyze both audio quality and real-time embeddability of the models.
 * [src](src/): The source code for the project, including implementations for training, evaluation, and the DDSP models.

This repository contains the code of the paper "Distilling DDSP: Exploring Real-Time Audio Generation on Embedded Systems.". 


### Get Started
It is recommended to use a virtual environment. We suggest using **conda**.

#### Conda with environment.yaml
*NOTE:* the venv create this way can contain also unnecesary packages.
```bash
# Clone project
git clone https://github.com/gregogiudici/distilling-ddsp.git
cd distilling-ddsp

# Create conda environment
conda env create -f environment.yaml
conda activate myenv
```

#### Conda + Pip
If you would like to create a minimal environment, instead create it from scratch with conda, and then install the other packages with pip.

```bash
# Clone project
git clone https://github.com/gregogiudici/distlling-ddsp.git
cd distilling-ddsp

# Create conda environment
conda create -n myenv python=3.9
conda activate myenv
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# Some care may be needed to make compatible the library versions of torch, torchaudio, etc

# Install requirements
pip install -r requirements.txt
```

## Dataset generation

We used two dataset:
1. [URMP](https://labsites.rochester.edu/air/projects/URMP.html) for flute, trumpet, and violin samples.
2. [Jazznet](https://tosiron.com/jazznet/) for piano samples.

Additional test files can be aggregated and used for resynthesis tasks.
Please check the `dataset` directory for advanced options to process and build a dataset.

**Quick start** - will extract and process violin, flute, trumpet, and piano data. Pitch and loudness are the input features of all DDSP models. We used [`torchcrepe`](https://github.com/maxrmorrison/torchcrepe) for pitch estimantion.

```bash
cd dataset
python create_data.py urmp.source_folder=/path/to/URMP/Dataset jazznet.source_folder=/path/to/Jazznet
```

## Training
#### Train with default configuration:
- model: hpn-full
- dataset: flute
- task: no_kd 
```bash
# Train with accelerator=auto (CPU, GPU, DDP, etc)
python src/train.py

# Train on CPU
python src/train.py trainer=cpu

# Train on GPU
python src/train.py trainer=gpu
```

#### Train a different model
```bash
# Train HpN-full (default) with Flute (default)
# dataset: sr=16kHz, fr=250
python src/train.py #model=hpn/full_hpn

# Train HpN-reduced
python src/train.py model=hpn/reduced_hpn

# Train DDX7-full
python src/train.py model=ddx7/full_ddx7_fmflute

# Train Wavetable-full
python src/train.py model=wavetable/full_wavetable

```
#### Train with a different instrument

```bash
# Train HpN-full (default) with Flute (default)
# dataset: sr=16kHz, fr=250
python src/train.py #data=flute_16000_250

# Train HpN-full (default) with Violin
python src/train.py data=violin_16000_250 

# Train HpN-full (default) with Trumpet
python src/train.py data=trumpet_16000_250
```

You can override any parameter from command-line, like this

```bash
python src/train.py trainer.max_epochs=5000 data.batch_size=64 data=violin_16000_250 
```



## Knowledge Distillation
To apply the Knowledge Distillation, you have to follow these steps:

### 1. Train TEACHER 

```bash
# Example:
#       dimension: full
#       model: HpN
#       instrument: flute
python src/train.py experiment=hpn/full_hpn_flute
```
### 2. Add TEACHER checkpoint to your example.env file
This is necessary for the distillation framework since you have to indicate where the pre-trained teacher model is located.
```bash
FULL_HPN_FLUTE="${ROOT}/logs/path/to/checkpoint/1999-12-31_00-00-00/checkpoints/epoch_000.ckpt"

```

### 3. Train STUDENT *without* Knowledge Distillation
```bash
# Experiment: 
#       dimension: reduced
#       model: HpN
#       instrument: flute
python src/train.py experiment=hpn/reduced_hpn_flute

```
### 4. MODIFY the yaml files:
Modify *task* in [train.yaml](/configs/train.yaml) from *no_kd* to *distillation*. 

Make sure the [distillation.yaml](/configs/task/distillation.yaml) has the correct teacher variables for both the architecture and the checkpoint path.

Check the configuration of the distillation methods in [distillation](configs/distillation/) folder.

### 5. Train STUDENT *with* Knowledge Distillation
```bash
# Experiment: 
#       dimension: reduced
#       model: HpN
#       instrument: flute
python src/distillation.py experiment=hpn/reduced_hpn_flute
```