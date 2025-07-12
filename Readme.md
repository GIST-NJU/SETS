# SETS: A Simple yet Effective Approach for DNN Test Selection

This folder serves as the replication package for the paper: “SETS: A Simple yet Effective Approach for DNN Test Selection”. It includes the source code of the proposed approach and baselines, all necessary scripts to reproduce the experiments, the data required for running the experiments, and the original experimental results.

---

## Folder Structure

```
.
├── Input_data
│   ├── Fault_clusters
│   ├── Pretrained_model
│   └── Retrain
├── Experiment_results
│   ├── RQ1
│   ├── RQ2&3
│   └── RQ4
├── Readme.md
├── requirements.txt
└── Source_code
```

## Requirements

The `requirements.txt` lists the dependencies required to run the Python code.
Run the following command to install dependency packages:

```
pip install -r requirements.txt
```

## SETS

The `Source_code\SETS.py`file contains the implementation of the **SETS** approach.
Run the following function to perform test selection for a given test selection problem:

```python
def sets(size, index, features, output_probability, uncertainty, diversity, a)
```

The input parameters include:

* `size`: test budget, i.e., number of test inputs that will be selected (*int*)
* `index`: indexes of all the test inputs (*list*)
* `features`: features of all the test inputs (*numpy.ndarray*)
* `output_probability`: output probabilities of all the test inputs (*numpy.ndarray*)
* `uncertainty`: metric for uncertainty evaluation, "maxp" or "gini" (*str*)
* `diversity`: metric for diversity evaluation, "gd" or "std" (*str*)
* `a`: the reduction coefficient, 3 by default (*int*)

The output will be a list of indexes of the selected test inputs and the execution time.

## Reproducing Experiments

### 1) Experimental Subjects

The experiment uses a total of six datasets, among which **MNIST**, **Fashion-MNIST**, and **CIFAR-10** can be directly loaded from `keras.datasets`. **Fruit-360**, **SVHN** and **TinyImageNet** datasets can be downloaded from the following links:

- **Fruit-360** : `https://github.com/fruits-360/fruits-360-100x100/tree/2f981c83e352a9d4c15fb8c886034c817052c80b`
- **SVHN** : `http://ufldl.stanford.edu/housenumbers/`
- **TinyImageNet**: `https://www.kaggle.com/c/tiny-imagenet/data?select=test.zip`

We provide all the pretrained DNN models in the `Pretrained_model` folder. You can also download them from the following links:

| Dataset        | Model   | Pretrained_download_link |
| :--------  | :-----  |:-----  |
| MNIST | LeNet1|`https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model`|
|MNIST | LeNet5|`https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model`|
| Fashion | LeNet4 |`https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model`|
| CIFAR-10 |12Conv |`https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model`|
| CIFAR-10 |ResNet20|`https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model`|
| SVHN |LeNet5 |`https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model`|
|Fruit-360  |ResNet50 |in the **pretrained_model** folder |
|TinyImageNet  |ResNet101 |`https://drive.google.com/drive/folders/1RLyQIcJ8qNqds9US-Oo2a0uQEL0t6kSZ`|

### 2) Baseline Approaches

#### DeepGD

The official replication package of is available at [DeepGD](https://github.com/ZOE-CA/DeepGD/tree/main).
Since the DeepGD approach in its original replication package is implemented in Jupyter Notebook (`.ipynb`), we have encapsulated it into the `Source_code/DeepGD.py` file. Note that the DeepGD approach involves randomness and may produce slightly different results in different runs.
Run the following function to perform test selection for a given test selection problem:

```python
def deepgd(size, index, gini_scores, features)
```
The input parameters include:

* `size`: test budget, i.e., number of test inputs that will be selected (*int*)
* `index`: indexes of all the test inputs (*list*)
* `gini_scores`: DeepGini uncertainty values of all the test inputs (*numpy.ndarray*)
* `features`: features of all the test inputs (*numpy.ndarray*)

The output will be a list of indexes of the selected test inputs and the execution time.

#### Random Selection (RS)

The `Source_code/RS.py` file provides the implementation of the RS approach.
Run the following function to perform test selection for a given test selection problem:

```python
def rs(size, index)
```
The input parameters include:

* `size`: test budget, i.e., number of test inputs that will be selected (int)
* `index`: indexes of all the test inputs (list)

The output will be a list of indexes of the selected test inputs.


### 3) Input Data Required

The `Input_data` directory contains pre-processed files for running test selection approaches, and reproducing the experiments. It includes:

* `/Pretrained_model`: this folder contains all the pretrained DNN models.
* `/Fault_clusters`: this folder contains contains pre-processed data for each subject, which will be used as the input for running the test selection approach and evaluate the results:
  *  `features_test.npy`: features of all the test inputs,
  *  `output_probability.npy`: output probabilities of all the test inputs,
  *   cluster_results.npy`: clustering label of each test input (for computing the Fault Detection Rate),
  *  `mis_index_test.npy`:  indexes of misclassified test inputs (for computing the Fault Detection Rate).
* `/Retrain`: this folder contains a test set \(T\) and a validation set \(V\) for each dataset used for evaluating DNN retraining performance (RQ4).

In order to generate the files in `/Fault_clusters`, use the following data processing scripts:

* Run `python feature.py` to generate `features_test.npy` and `output_probability.npy` files. This involves loading the dataset and model, performing predictions to obtain the output probabilities (the prediction time is recorded at the same time), and extracting features using a VGG16 model.
* Run `python cluster.py` to generate `cluster_results.npy` and `mis_index_test.npy` files. Note that the [DBSCAN clustering algorithm](https://github.com/zohreh-aaa/DNN-Testing) involves randomness and may produce slightly different results in different runs.

### 4) Experiment Execution

After obtaining all the required input data in `Input_data` (either by using the pre-generated data directly, or running data processing scripts to generate such data), run the following command to perform the experiments of each research question:

- **RQ1 (Configuration)**: run `python exp_1.py [data_path]`. This will run SETS with different combinations of uncertainty and diversity metrics, and with different values of the reduction coefficient on all subjects. 
- **RQ2&3 (Efficiency and Effectiveness)**:  
  Run `python exp_2_3.py [approach] [n] [data_path] [output_path]` to apply a specific approach to perform test selection on all subjects.  
  - `[approach]`: The test selection method to use. Available options are `sets`, `deepgd`, and `rs`.  
  - `[n]`: The number of times to repeat the experiment (e.g., 30).  
  - `[data_path]`: The path to the `Fault_clusters` folder.  
  - `[output_path]`: The path to save the result files.  

  You will get result files that are consistent with those in the `Experiment_results` folder.

- **RQ4 (DNN Enhancement)**: run `python exp_4.py [data_path]`. You will get the subsets selected by SETS and DeepGD on the test set T and you need to pass them into the corresponding retraining scripts (`retrain_four.py`, `retrain_fruit.py`, and `retrain_tiny.py`) for model retraining. The accuracy of the retrained model will then be evaluated.

In the above commands, `[data_path]` is the actual path where your `Fault_clusters` folder is located.

### 5) Experiment Results

The `Experiment_results` folder provides all raw experimental results reported in the paper:

- **RQ1 (Configuration):** We provide the Fault Detection Rate (FDR) of SETS with different metric combinations in the `Metric_combination` folder and the FDR and time cost of SETS with different reduction coefficients in the `reduction_coefficient` folder (Results of 10 repeated runs).
- **RQ2&3 (Efficiency and Effectiveness):** We provide the FDR and execution time of SETS and baseline approaches in their respective folders (Results of 30 repeated runs).
- **RQ4 (DNN Enhancement):** We provide the retraining results of SETS and DeepGD in the `Retrain_results` folder, as well as the selected subsets in their respective folders (Results of 5 repeated runs).

