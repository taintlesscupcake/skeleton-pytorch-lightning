# PyTorch Default Code with PyTorch Lightning

This repository contains a Default code for training and testing deep learning models using PyTorch Lightning. The Default code is organized into several modules, including `module.py`, `model.py`, and `dataloader.py`. You can easily customize this code to suit your specific use cases.
<!---
[![한국어](https://img.shields.io/badge/한국어-Readme-blue)](./README_KO.md)
-->
[Korean README](./README_KO.md)

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

## Requirements

- Python 3.7 or later
- PyTorch 1.8.0 or later
- PyTorch Lightning 1.4.0 or later
- torchvision (optional, if needed for your dataset)


## Project Structure

The project contains the following files:

- `train.py`: The main script to train your model.
- `test.py`: The main script to test your model.
- `module.py`: Contains the `DefaultModule` and `DefaultDataModule` classes.
- `model.py`: Contains the `DefaultModel` class.
- `dataloader.py`: Contains the `DefaultDataset` class.
- `train.sh`: A shell script to run the training script.
- `test.sh`: A shell script to run the testing script.

## Usage

First, clone this repository:

```bash
git clone https://github.com/taintlesscupcake/Default-pytorch-lightning.git
cd Default-pytorch-lightning
```

To train your model, run the following command:

```bash
./train.sh
# use chmod +x train.sh if you get a permission error
```


To test your model, run the following command:

```bash
./test.sh
# use chmod +x test.sh if you get a permission error
```

## Customization

To customize the code for your specific use case, follow these steps:

1. **Update the `DefaultModel` in `model.py`**: Define your own deep learning model architecture by modifying the `DefaultModel` class.

2. **Update the `DefaultDataset` in `dataloader.py`**: Customize the `DefaultDataset` class to load your specific dataset.

3. **Update the `DefaultModule` in `module.py`**: Modify the `DefaultModule` class to include your loss function, optimization algorithm, and learning rate scheduling.

4. **Update the `DefaultDataModule` in `module.py`**: Customize the `DefaultDataModule` class to set your data loading pipeline, including data augmentation and data splitting.

5. **Modify `train.py` and `test.py`**: Adjust the arguments and settings in `train.py` and `test.py` to match your customized classes and desired training/testing configurations.

## License

This project is licensed under the [MIT License](LICENSE).
