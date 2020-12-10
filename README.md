# EPFL Machine Learning: Calibrate a model of OTC markets.

__`Team`__: arbitrageurs

__`Team Members`__: Qais Humeid, Kim Ki Beom, Louis Leclair

Before starting our project you must have the following libraries installed on your computer:

run the ```pip install -r requirements.txt``` command to have all the following library install

1. Numpy
2. pytorch
3. matplotlib
4. pandas
5. xgboost
6. sklearn

- Run the following script
- download the git 
- run python script `run.py`

All the details are in the jupyter notebook where you can find all our plots, our data manipulations and etc...

## Structure of the project 
```
├── MLProject-Oct2020.pdf
├── README.md
├── datas
│   ├── all_datas.csv
│   ├── data /
│   └── original_data /
├── result
└── src
    ├── helper.py
    └── project2.ipynb

```

## Modules of the project

### `Datas`:
Contains all our data files from the original data in .dat in`original_data`to the convert data in csv in`data`to the concatenation of all the data files in`all_datas.csv`.

### `Helper.py`: 
Contains some helper functions to ease the comprehension and the density of the code in the notebook.

#### `Loss Functions`
- __`MSE`__: Compute the mean square error of 2 tensors.
- __`MAE`__: Compute the mean absolute error of 2 tensors.

#### `Cross Validation functions`
Helpers functions  __`build_k_indices`__, __`cross_validation`__. Which help us to find the best input and create some randomness to better training and testing.


