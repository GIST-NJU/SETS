from numpy import argmax
import random
import seaborn as sbn
from numpy.random import rand, randn
from scipy.linalg import qr
from numpy import ones
from scipy import stats
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
import  array
import math
import pickle
# from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from sklearn import linear_model
import sklearn
# from tabulate import tabulate
from math import sqrt
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA as sklearnPCA
import copy
import time
from keras import backend as K
import argparse
import shutil
import warnings
import keras.backend as KeyboardInterrupt
from keras.regularizers import l2
from keras.models import load_model, Model
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.datasets import mnist, cifar10 , fashion_mnist, cifar100
import sys
from numpy.core.defchararray import array
import os
sys.path.append('..')
import ast
import numpy as np
from scipy.spatial.distance import cdist
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.sampling import Sampling

import time

import multiprocessing
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import string
import numpy as np
from pymoo.factory import get_sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import string
import numpy as np
from pymoo.factory import get_sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import matplotlib.pyplot as plt
import numpy as np
from pymoo.factory import get_algorithm
from pymoo.visualization.scatter import Scatter

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import string
import numpy as np
from pymoo.factory import get_sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import random
import numpy as np
from pymoo.core.callback import Callback
import cProfile
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tensorflow.keras.optimizers import SGD, Adam,Adadelta,Adagrad
from numpy.random.mtrand import sample
from sklearn.utils import shuffle

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def dataset(arg, model_name):
  CLIP_MIN = -0.5
  CLIP_MAX = 0.5
  flag = True

  if arg=="mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # ##Model
    if model_name=="LeNet1":
        model =  load_model(str(data_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
    if model_name=="LeNet5":
        model =  load_model(str(data_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")

    y_test = to_categorical(y_test, 10)
    y_test=np.argmax(y_test, axis=1)
    y_train = to_categorical(y_train, 10)

  if arg=="Fashion_mnist":
    # load dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    if model_name=="LeNet4":
      model =  load_model(str(data_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
      y_test = to_categorical(y_test, 10)
      y_test=np.argmax(y_test, axis=1)
      y_train = to_categorical(y_train, 10)


  if arg=="SVHN":
    train_raw = loadmat(str(data_path)+'/Data/train_32x32.mat')
    test_raw = loadmat(str(data_path)+'/Data/test_32x32.mat')
    x_train = np.array(train_raw['X'])
    x_test = np.array(test_raw['X'])
    y_train = train_raw['y']
    y_test = test_raw['y']
    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)
    x_test= x_test.reshape (-1,32,32,3)
    x_train= x_train.reshape (-1,32,32,3)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    if model_name=="LeNet5":
      model =  load_model(str(data_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
      lb = LabelBinarizer()
      y_train = lb.fit_transform(y_train)
      y_test = lb.fit_transform(y_test)
      y_test=np.argmax(y_test, axis=1)

  if arg=="cifar10":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    if model_name=="12Conv":
      model =  load_model(str(data_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
    if model_name=="ResNet20":
      model =  load_model(str(data_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")

    y_test = to_categorical(y_test, 10)
    y_test=np.argmax(y_test, axis=1)
    y_train = to_categorical(y_train, 10)
  
  if arg=="fruit360":
    x_train = np.load(str(data_path)+"/Fruit360_ResNet50/fruit_x_train_origin.npy")
    y_train = np.load(str(data_path)+"/Fruit360_ResNet50/fruit_y_train.npy")
    x_test = np.load(str(data_path)+"/Fruit360_ResNet50/fruit_x_test_origin.npy")
    y_test = np.load(str(data_path)+"/Fruit360_ResNet50/fruit_y_test.npy")
    model = load_model(str(data_path)+"/Fruit360_ResNet50/fruit_resnet2.h5")
    flag = False
  
  if flag:
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

  return x_train, y_train, x_test, y_test, model


data_model_pairs = [
    #("mnist", "LeNet1"),
    #("mnist", "LeNet5"),
    #("cifar10", "12Conv"),
    #("cifar10", "ResNet20"),
    #("Fashion_mnist", "LeNet4"),
    #("SVHN", "LeNet5"),
    ("fruit360","resnet50"),
    #("tinyimagenet","resnet101")
]

# model retrain parameters all the same
#opt = tf.keras.optimizers.legacy.Adadelta(learning_rate=0.005)
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001)
Epi=0
final=[]
ep = 30
bat =100
optim=opt
Ac=[]
indx=0

classes = 10

if __name__ == '__main__':

  import sys
  output_path = sys.argv[1]
  data_path = sys.argv[2]

  for data_name, model_name in data_model_pairs:
      print(f"Dataset: {data_name}, Model: {model_name}")
      x_train, y_train, x_test, y_test, model = dataset(data_name, model_name)

      #print(y_test)
      f_name = f"{output_dir}/{data_name}_{model_name}.pkl"
      with open(f_name, 'rb') as f: #get the testing set T
        loaded_indices = pickle.load(f)
      elements = list(range(23619))
      v_set = list(set(elements) - set(loaded_indices))
      v_test = x_test[v_set]
      vy_test = y_test[v_set]


      if data_name == "fruit360":
        classes = 141
      if data_name == "tinyimagenet":
        classes = 200

      #y_pred = model(x_test) first 6 subject
      #y_pred = model.predict(x_test)
      y_pred = model.predict(v_test)
      y_pred = np.argmax(y_pred, axis=1)

      #f1_score(y_test, y_pred, average="weighted")

      acc_ori = accuracy_score(vy_test, y_pred)
      print(f"Original Model Accuracy: {acc_ori:.4f}")
      #y_test1 = to_categorical(y_test, classes)
      #y_train1 = to_categorical(y_train,classes) #6 subject no need

      #print("y_test1")
      #print(y_test1)
      #print("y_test")
      #print(y_test)
      #print("y_train")
      #print(y_train)
      #print("y_train1")
      #print(y_train1)

      x_tr1 = copy.deepcopy(x_train)
      y_tr1 = copy.deepcopy(y_train)
      #y_tr1 = copy.deepcopy(y_train1)

      print("Start Retraining!")


      for size in [500]:

        #DeepGD
        file_name = str(data_path)+f"/DeepGD/Retrain/{data_name}_{model_name}_{size}.txt"
        with open(file_name, "r") as f:
          lines = f.readlines()

        subset_list = []
        in_subset_list = False
        subset_data = ""
        for line in lines:
          if "Subset List:" in line:
            in_subset_list = True
            continue

          if in_subset_list:
            subset_data += line.strip() + " "
            if "]" in line:
              subset_data = subset_data.replace("[", "").replace("]", "").strip()
              subset_data_cleaned = subset_data.replace(",", "")
              subset_list.append(list(map(int, subset_data_cleaned.split())))
              subset_data = ""

        acc_re_list = []
        acc_imp_list = []
        t_id = random.sample(range(len(x_train)),10000) #5000 是正的，至少ETS有效
        for subset in subset_list[:5]:


          x_newtr = np.concatenate((x_tr1, x_test[subset]), axis=0)
          y_newtr = np.concatenate((y_tr1, y_test[subset]), axis=0)

          #model = load_model(str(data_path) + "/Retraining/Org_model/model_" + str(data_name) + "_" + str(model_name) + ".h5")
          model = load_model(str(data_path)+"/Pretrained_model/fruit_resnet2.h5")
          #model.compile(optimizer=optim, loss='categorical_crossentropy')
          #base_model = model.get_layer('resnet50')
          #for layer in base_model.layers[-4:]:
          #  layer.trainable = True
          model.compile(optimizer=optim, loss='sparse_categorical_crossentropy')



          x, y = shuffle(x_newtr, y_newtr)
          #print(x.shape)
          #print(y.shape)
          #x, y = shuffle(x_test[subset], y_test[subset])
          v_id = random.sample(range(len(x_test)), 5000)

          model.fit(x, y, epochs=ep, batch_size=bat, validation_data=(x_test[v_id], y_test[v_id]), verbose=0)
          #model.fit(x, y, epochs=ep, batch_size=bat, validation_data=(x_test[v_id], y_test[v_id]), verbose=0)
          #ypre1 = model(x_test)
          ypre1 = model.predict(v_test)
          ypre1 = np.argmax(ypre1, axis=1)
          #print(ypre1)
          acc_re = accuracy_score(vy_test, ypre1)
          acc_re_list.append(acc_re)
          print(f"Retrained Model Accuracy: {acc_re:.4f}")
          acc_imp = acc_re - acc_ori
          #print(f"Accuracy Improvement: {acc_imp:.4f}")
          acc_imp_list.append(acc_imp)

        print(f"average accuracy imp DeepGD {size}: {(sum(acc_imp_list)/len(acc_imp_list)):.4f}")



        #SETS
        file_name = str(data_path)+f"/SETS/Retrain/{data_name}_{model_name}_{size}.pkl"
        with open(file_name, 'rb') as f:
          sub_ets = pickle.load(f)

        acc_re_ets_list = []
        acc_imp_ets_list = []
        for i in range(5):
          x_newtr_ets = np.concatenate((x_tr1, x_test[subset]), axis=0)
          y_newtr_ets = np.concatenate((y_tr1, y_test[subset]), axis=0)
          #y_newtr_ets = np.concatenate((y_tr1, y_test[subset]), axis=0)

          #model = load_model(str(data_path) + "/Retraining/Org_model/model_" + str(data_name) + "_" + str(model_name) + ".h5")
          model = load_model(str(data_path)+"/Pretrained_model/fruit_resnet2.h5")
          #model.compile(optimizer=optim, loss='categorical_crossentropy')
          model.compile(optimizer=optim, loss='sparse_categorical_crossentropy')

          x, y = shuffle(x_newtr_ets, y_newtr_ets)
          v_id = random.sample(range(len(x_test)), 5000)

          model.fit(x, y, epochs=ep, batch_size=bat, validation_data=(x_test[v_id], y_test[v_id]), verbose=0)
          #model.fit(x, y, epochs=ep, batch_size=bat, validation_data=(x_test[v_id], y_test[v_id]), verbose=0)
          #model.fit(x, y, epochs=ep, batch_size=bat, validation_data=(x_test, y_test1), verbose=0)
          #ypre1 = model(x_test)
          ypre1 = model.predict(v_test)
          ypre1 = np.argmax(ypre1, axis=1)
          acc_re_ets = accuracy_score(vy_test, ypre1)
          acc_re_ets_list.append(acc_re_ets)
          print(f"Retrained Model Accuracy SETS: {acc_re_ets:.4f}")
          acc_imp_ets = acc_re_ets - acc_ori
          #print(f"Accuracy Improvement ETS: {acc_imp_ets:.4f}")
          acc_imp_ets_list.append(acc_imp_ets)

        print(f"average accuracy imp SETS {size}: {(sum(acc_imp_ets_list)/len(acc_imp_ets_list)):.4f}")

        file_path = str(output_path)+f"/new_retrain_res/{data_name}_{model_name}_{size}_3.txt"
        with open(file_path, "w") as f:
          f.write(f"Original Model Accuracy: {acc_ori:.4f}\n")

          f.write("DeepGD Retraining\n")
          f.write("Acc Re List:\n")
          f.write("\n".join([f"{x:.4f}" for x in acc_re_list]) + "\n")
          f.write("Acc Imp List:\n")
          f.write("\n".join([f"{x:.4f}" for x in acc_imp_list]) + "\n")

          f.write("SETS Retraining\n")
          f.write("Acc Re List:\n")
          f.write("\n".join([f"{x:.4f}" for x in acc_re_ets_list]) + "\n")

          f.write("Acc Imp List:\n")
          f.write("\n".join([f"{x:.4f}" for x in acc_imp_ets_list]) + "\n")

        print("save successfully!")



