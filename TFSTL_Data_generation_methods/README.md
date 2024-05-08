# TFSTL Model Data Generation Guide

This README provides a step-by-step guide on generating the necessary data inputs for the TFSTL model. Follow these steps carefully to ensure the correct setup and execution.

## Prerequisites

Before you begin, ensure that you have Python installed on your machine (Python 3.8 or newer is recommended). This guide also assumes that you have basic knowledge of Python programming and handling of CSV files.

## Installation Steps

### Step 1: Install the GraphEmbedding Package

You need to install the `GraphEmbedding` package which is crucial for the data preprocessing steps. You can install it from the GitHub repository or use the provided package if available. Run the following command in your terminal:

```bash
pip install git+https://github.com/shenweichen/graphembedding.git
# Or if a local version is provided:
cd GraphEmbedding
python setup.py install
```

### Step 2: Check Your Data Folder

Ensure that you have a CSV file named `import_data.csv` in your data folder. This file should contain a matrix where each row represents a date and each column represents a commodity. The dimensions of the matrix should be N (number of dates) by M (number of commodities).

### Step 3: Modify and Run the Preprocessing Script

1. Open the `TFSTL_data_preprocessing_script.py` file.
2. Modify the `directory` variable to the path where your `import_data.csv` file is located.
3. Ensure that the `GraphEmbedding` package is properly installed and can be imported without issues.
4. Run the script by executing the following command in your terminal:

```bash
python TFSTL_data_preprocessing_script.py
```

You can initially try running the script using the example data `data/MYDATA/new_import` or `data/MYDATA/new_export` provided by the author, which includes all necessary files.

### Step 4: Generate Required Files and Configure the Model

Once all required files are generated, you can proceed to modify the TFSTL's configuration file to run the model with your data. Pay careful attention to all path settings in the configuration to avoid unnecessary errors.

## Additional Notes

- Ensure that all paths in the scripts and configuration files are correct and accessible.
- Review the output files for correctness after running the preprocessing script.

If you encounter any issues during the installation or data generation process, please review the steps and ensure all prerequisites are met.
