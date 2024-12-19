# Dataset generation

1. Please modify in `create_data.yaml` file path to the `source_foulder` directory for both the URMP dataset and Jazznet dataset. Alternatevely, add URMP_PATH and JAZZNET_PATH variables in your .env file.

2. If you wish to generate a testset using audio from a different source, you can populate the secion `testset` of the `create_data.yaml` file
with a path to a location that contains directories with instrument names. Each directory should contain `.wav` files of the corresponding instrument.
Also, make sure you run `create_data.py` with the option `process_testset=True`.

3. You can also use the `testset` section to process external files into continuous test sets, i.e. not chopped. Just make sure the option `testset.contiguous`
is set to `True`. Then, load the additional testset when training with the option `load_additional_testset: True` located in `recipes/config.yaml`.