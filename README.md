# EPFL Machine Learning: Calibrate a model of OTC markets.

__`Team`__: arbitrageurs

__`Team Members`__: Qais Humeid, Kim Ki Beom, Louis Leclair

After downloading the git.
To start our project you must have the following libraries installed on your computer:

1. Numpy
2. matplotlib
3. pandas
4. xgboost
5. scipy
6. [pytorch](https://pytorch.org) (see install pytorch section)
7. [sklearn](https://scikit-learn.org/stable/install.html)

To install them, run the ```pip install -r requirements.txt``` command to have Numpym matplotlib, pandas and xgboost installed and for pytorch and sklearn you have to click on the link above and follow the instruction on the different websites.

All the details are in the jupyter notebook file ``project2.ipynb``where you can find all our plots, our data manipulations, our trials and results. The code inside is comment to help you understand our thoughts and reflexion.

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

### `Helper.py`:  Contains some helper functions to ease the comprehension and the density of the code in the notebook.

#### `Loss Functions`
- __`MSE`__: Compute the mean square error of 2 tensors.
- __`MAE`__: Compute the mean absolute error of 2 tensors.

#### `Cross Validation functions`
Helpers functions  __`build_k_indices`__, __`cross_validation`__. Which help us to find the best input and create some randomness to better training and testing. As well as __`cross_correlation_regression`__ and __`cross_correlation_boost`__ which are functions used to do linear regression with different models and loss functions.

#### `Data Analysis`
We just have the function __`show_heatmap`__ which helped us to find the correlation between each features of the given dataset.

#### `Neural Network`

We only have the __`train`__ method inside this part which is almost the same function as the one define in lab10 of this course with some minor modifications.

### `Project2.ipynd`

All our thoughts and reflexions are inside this file. The plan of the notebook is the following:

- Import
- Data analysis of a sample of size `threshold` with plots.
- Linear regression part with cross correlation to have different results. The tested models are in this order `XGBoost`, `AdaBoost`, `Lasso`, `Linear Regression` the basic one, `SGDRegressor`.
- Neural Network part, where we have a `CNN`, a `Fully Connected Neural Network` and the part where we solve the problem asked where we find a __Θ*__ with the help of __`scipy`__ to find the minimum of the function.

