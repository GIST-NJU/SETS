from numpy import linalg as LA
import time
import numpy as np
import pickle
import copy
import sys

from RS import rs

# This function is adapted from the implementation in:
# https://github.com/ZOE-CA/DeepGD
def faults(sample,mis_ind_test,Clustering_labels):
  i=0
  pos=0
  neg=0
  i=0
  cluster_lab=[]
  nn=-1
  for l in sample:
    if l in list(mis_ind_test):
      neg=neg+1
      # print("index mis",l)
      ind=list(mis_ind_test).index(l)
      if (Clustering_labels[ind]>-1):
        cluster_lab.append(Clustering_labels[ind])
      if (Clustering_labels[ind]==-1):
        cluster_lab.append(nn)
        nn=nn-1
    else:
      pos=pos+1

    # i=i+1
  faults_n=len(list(set(cluster_lab)))

  cluster_1noisy=copy.deepcopy(cluster_lab)
  for i in range(len(cluster_1noisy)):
   if cluster_1noisy[i] <=-1:
     cluster_1noisy[i]=-1
  faults_1noisy=len(list(set(cluster_1noisy)))
  return faults_n,faults_1noisy, neg


data_model_pairs = [
    ("mnist", "LeNet1"),
    ("mnist", "LeNet5"),
    ("cifar10", "12Conv"),
    ("cifar10", "ResNet20"),
    ("Fashion_mnist", "LeNet4"),
    ("SVHN", "LeNet5"),
    ("Fruit360","ResNet50"),
    ("TinyImageNet", "ResNet101")
]

if len(sys.argv) < 2:
    print("Usage: python exp_3_rs.py <data_path>")
    sys.exit(1)

data_path = sys.argv[1]


for data_name, model_name in data_model_pairs:
    print(f"Dataset: {data_name}, Model: {model_name}")
    if data_name == "TinyImageNet":
        with open(f'{data_path}/{data_name}_{model_name}/cluster_results.pkl', 'rb') as f:
            Clustering_labels = pickle.load(f)
        with open(f'{data_path}/{data_name}_{model_name}/mis_index_test.pkl', 'rb') as f:
            val_mis_data = pickle.load(f)
        mis_ids_test = [item[2] for item in val_mis_data]
    else:
        Clustering_labels = np.load(f'{data_path}/{data_name}_{model_name}/cluster_results.npy')
        mis_ids_test = np.load(f'{data_path}/{data_name}_{model_name}/mis_index_test.npy')

    output_probability = np.load(f'{data_path}/{data_name}_{model_name}/output_probability.npy')
    features_test = np.load(f'{data_path}/{data_name}_{model_name}/features_test.npy')

    if data_name == "Fruit360":
        mis_ids_test = mis_ids_test[0]


    total_faults = len(set(Clustering_labels)) - 1

    noisy_index = []
    for i in range(len(mis_ids_test)):
        if Clustering_labels[i] == -1:
            noisy_index.append(mis_ids_test[i])
    sett = list(range(0, len(output_probability)))
    index_withoutnoisy = set(sett) - set(noisy_index)


    for size in [100,300,500]:
        print(size)
        selected_subset = rs(size,index_withoutnoisy)
        _, find_faults, _ = faults(selected_subset, mis_ids_test, Clustering_labels)
        fdr = find_faults / min(size, total_faults)
        print("FDR:", fdr)