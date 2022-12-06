# EXPERIMENTAL Complement Sparsification(CS) Simulation using Flower and TensorFlow/Keras

Code for paper "Complement Sparsification: Low-Overhead Model Pruning for Federated Learning" in AAAI 2023: https://www.researchgate.net/publication/366027404_Complement_Sparsification_Low-Overhead_Model_Pruning_for_Federated_Learning

This project uses Flower and TensorFlow/Keras to simulate CS.

## Dataset
This project uses LEAF dataset from https://github.com/TalwalkarLab/leaf. 
Please download FEMNIST dataset following the instructions in LEAF. To fully reproduce the results, please be noted that we use the full dataset with non-iid setting. i.e. ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample.
Please run generate_flower_data.py to generate the data for Flower from LEAF. Make sure to change "out_path", "train_data_dir", "test_data_dir" in the script, and be free to change "split". 

## Running the project (via Poetry)
Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` (the modern alternative to `requirements.txt`). I recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that it works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go! 
Edit the script "fem_sim.py" for your own settings.
Please be free to try different hyper-parameters at the beginning of the scrip.

To run the experiments. Please run the following command. 
```bash
poetry run python3 fem_sim.py
```

Please be free to ask me if you have any questions.