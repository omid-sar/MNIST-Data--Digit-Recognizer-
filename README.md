
# MNIST Digit Recognizer with Advanced Techniques

## Goal

This project aims to create a high-performance Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. We explore advanced strategies such as data augmentation, hyperparameter tuning using Keras Tuner, and model ensembling to enhance the model's performance and robustness.

## Overview

The project is structured around implementing and evaluating a CNN based on the LeNet architecture, optimizing it through various techniques, and assessing its effectiveness in classifying handwritten digits from the MNIST dataset.

## Features

- **Data Augmentation**: The augmentation increases the diversity of the training dataset, enhancing the model's generalization ability by applying transformations like rotation and shifting.
- **Hyperparameter Optimization**: Keras Tuner is employed for tuning the model's hyperparameters, enabling the selection of the optimal configuration for improved performance.
- **Ensembling**: Multiple models are combined to improve overall performance and robustness by leveraging their individual strengths.
- **LeNet Architecture**: The foundational architecture for constructing the CNN is the LeNet model.

## Project Organization

```
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                  <- predictions. Includes scripts for model evaluation and ensembling.
    │   │   ├── predict_model.py
    │   │   ├── train_model.py
    │   │   └── ensemble_models.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    └── tox.ini            <- Tox file with settings for running tox; see tox.readthedocs.io
```

## Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Keras Tuner
- NumPy
- pandas
- Matplotlib
- scikit-learn
- pickle

## Setup and Installation

### Using `requirements.txt`

1. Clone the repository:
 ```sh
   git clone https://github.com/omid-sar/MNIST-Data--Digit-Recognizer-.git
   cd MNIST-Data--Digit-Recognizer-
   ```

2. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```

### Using `environment.yml` (Conda Environment)

1. Clone the repository:
   ```sh
   git clone https://github.com/omid-sar/MNIST-Data--Digit-Recognizer-.git
   cd MNIST-Data--Digit-Recognizer-
   ```

2. Create a new conda environment from the `environment.yml` file:
   ```sh
   conda env create -f environment.yml
   ```

3. Activate the newly created conda environment:
   ```sh
   conda activate mnist-env
   ```


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for details.