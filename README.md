# TFSTL
This is a PyTorch implementation of the paper: Trade Forecasting via Efficient Multi-commodity STL Decomposition based Neural Networks.


## Requirements

The model is implemented using Python3 with dependencies specified in `requirements.txt`.
Install the dependencies using the following command:

   ```bash
   pip install -r requirements.txt
   ```



## Data Preparation

The data for this project is located in the `data/MYDATA` folder. Within this folder, there are two subfolders organizing the dataset: 
- The `import` folder contains data related to imports.
- The `export` folder contains data related to exports.

### Data Description

| Imported Commodity                                     | Abbreviation | Exported Commodity                             | Abbreviation |
|--------------------------------------------------------|--------------|------------------------------------------------|--------------|
| Unwrought Copper and Copper Materials                  | Cu           | Other Agricultural Products                    | Agri         |
| Natural Gas                                            | NG           | Electrical Equipment                           | EE           |
| Metal Ores and Concentrates                            | Metal        | Other                                          | Other        |
| Automatic Data Processing Equipment and Its Parts      | ADPE         | Electrical Appliances                          | EA           |
| Integrated Circuits                                    | IC           | Refined Oil                                    | RO           |
| Grain                                                  | Grain        | Plastic Products                               | PP           |
| Other Agricultural Products                            | Agri         | Furniture and its Parts                        | FP           |
| Coal and Lignite                                       | Coal         | Steel                                          | Steel        |
| Other                                                  | Other        | Clothing, Accessories and Textile              | Textile      |
| Pulp, Paper and Paper Products                         | Pulp         | Automatic Data Processing Equipment and its Parts | ADPEP      |
| Automobiles and Parts                                  | Auto         | Grain                                          | Grain        |
| Primary Shaped Plastic                                 | Plastic      | Integrated Circuit                             | IC           |
| Crude Oil                                              | Crude        | Automobiles and Parts                          | Auto         |

### Before Training
- Make sure to create folders for 'cpt' and 'log'.
- Additionally, please remember to modify the paths in the 'config' file and 'train.py(train_meta_export.py or train_meta_import.py)' to your custom paths as needed.

## Model Training

### Import
```
python train_meta_import.py --config ./config/MYDATA_meta_import.conf
```

### Export
```
python train_meta_export.py --config ./config/MYDATA_meta_export.conf
```

## Remaining File Description

- config folder: contains configuration files for training.
- data folder: contains data for training.
- mymodel folder: contains the model TFSTL.
- lib folder: This folder contains a variety of utility functions essential for the project. Key functions included are:
  - stl_decomposition: Used for Seasonal-Trend decomposition using Loess, this function helps in breaking down a time series into seasonal, trend, and residual components, which is crucial for analyzing time-series data effectively.
  - Dataloader: This utility is responsible for loading and preprocessing data.
  - seq2instance: It facilitates the division of data into training and testing sets or other necessary partitions.
- log folder: contains the log files.
- output folder: Contains output files with 12-period forecasts for imports and exports, each in two steps. Every file has a shape of [12, 13], representing 12 forecast periods across 13 different categories of goods for both imports and exports. **The feature order is the same as in the Data Description.**
- runs folder: This folder contains the TensorBoard files, which include valuable information such as the loss functions for the import and export training. 

    - For Export Training:
    ```bash
    tensorboard --logdir='runs/STL_export'
    ```

    - For Import Training:
    ```bash
    tensorboard --logdir='runs/STL_import'
    ```
- cpt folder: This folder stores all the parameters of the import and export models, facilitating direct reuse and further training.
- .vscode folder: This folder includes configuration files for the Visual Studio Code (VSCode) Integrated Development Environment (IDE). These files are particularly useful when debugging in VSCode. Be aware that you may need to modify the file paths in these configurations to match your custom project paths.
