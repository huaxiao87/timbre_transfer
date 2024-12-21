# Embeddability Evaluation of DDSP Models on Raspberry Pi

This folder contains the code required for evaluating the Embeddability of various DDSP models (Harmonic-plus-Noise, DDX7, and Wavetable) on a Raspberry Pi embedded system.

## Structure

- **configs/**: Contains the configuration files of each DDSP model both in "full" and "reduced" version.
- **DDSP/**: Contains the code implementing the various DDSP models.
- **inference.py**: Script for analyzing the Real-Time Factor.
- **analysis.ipynb**: Jupyter notebook for analyzing the models.

## Instructions

1. Copy the `embeddability` folder into your Raspberry Pi. We suggest using `scp` for this purpose:
    ```sh
    scp -r embeddability pi@<raspberry_pi_ip>:/path/to/destination/
    ```

2. The `rpi_myenv.yml` file contains the virtual environment configuration for the analysis. We suggest using [MiniForge](https://github.com/conda-forge/miniforge) to set up the environment:
    ```sh
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
    bash Miniforge3-Linux-aarch64.sh
    conda env create -f rpi_myenv.yml
    conda activate myenv
    ```

3. Analyze **Real-Time Factor** with [inference.py](./inference.py) choosing the corresponding json file in `config` folder
    ```sh
    # For example
    python inference.py --config configs/hpn/hpn_full.json
    ```
    (Repeat the process for each model)

4. Analyze **FLOPs** and **Memory Usage** of each model using [analyze.ipynb](./analysis.ipynb)
