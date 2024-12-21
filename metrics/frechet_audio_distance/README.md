# Frechet Audio Distance (FAD)

This repository contains the code to calculate the Frechet Audio Distance (FAD).

## Usage

To calculate the FAD, use the `calculate_fad.py` script with the appropriate configuration specified in the `fad.yaml` config file.

### Steps

1. Ensure you have the necessary dependencies installed ([frechet_audio_distance](https://github.com/gudgud96/frechet-audio-distance)).
2. Configure the `fad.yaml` file with the required parameters.
3. Run the `calculate_fad.py` script:

```bash
python calculate_fad.py
```

## Configuration

The `fad.yaml` file should contain the necessary configuration settings for the FAD calculation. Make sure to update this file with the correct paths and parameters before running the script. 

NOTE: fad.yaml uses the environment variables present in [example.env](../../example.env)

