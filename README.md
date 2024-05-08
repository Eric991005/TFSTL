# TFSTL
This is a PyTorch implementation of the paper: Trade Forecasting via Efficient Multi-commodity STL Decomposition based Neural Networks.

[05/2024] TFSTL is currently under peer review by the International Journal of Forecasting.

## Requirements

The model is implemented using Python3 with dependencies specified in `requirements.txt`.
Install the dependencies using the following command:

   ```bash
   pip install -r requirements.txt
   ```



## Data Preparation

The data for the TFSTL project is organized as follows:

1. **Data Location and Structure**: The dataset is located within the `data/MYDATA` folder. This folder contains two subfolders:
   - `new_import`: Contains data related to imports.
   - `new_export`: Contains data related to exports.

2. **Original Data Files**: The initial datasets are stored in two CSV files:
   - `import_new.csv`: Contains the import data.
   - `export_new.csv`: Contains the export data.

3. **Generating Additional Input Data**:
   - If you wish to generate additional input data for the model, you can refer to the methods outlined in the `TFSTL_Data_generation_methods` folder.
   - To process and prepare all the model input data, you should use the `TFSTL_data_preprocessing_script.py`.

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
- result_check_and_visualize folder: This folder contains the script to check the results and visualize the output.
