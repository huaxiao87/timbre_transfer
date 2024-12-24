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

 * [configs](configs/): Contains configuration files for the project, utilizing Hydra for managing configurations.
 * [dataset](dataset/): Includes code for generating and preprocessing datasets required for training and evaluation.
 * [docs](docs/): Documentation for the project's webpage, providing detailed information and guides.
 * [metrics](metrics/): Contains metrics used to analyze both audio quality and real-time embeddability of the models.
 * [src](src/): The source code for the project, including implementations for training, evaluation, and real-time audio synthesis.

This repository contains the implementation of the paper "Distilling DDSP: Exploring Real-Time Audio Generation on Embedded Systems." The project focuses on distilling the Differentiable Digital Signal Processing (DDSP) framework to enable real-time audio generation on resource-constrained embedded systems. 


### Get Started
It is recommended to install this repo on a virtual environment. We suggest using **conda**.

#### Conda with environment.yaml
*NOTE:* the venv create this way can contain also unnecesary packages.
```bash
# Clone project
git clone https://github.com/gregogiudici/distlling-ddsp.git
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

