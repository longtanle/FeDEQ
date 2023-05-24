import os
import random
import time

import numpy as np
import scipy.io
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from torchvision import datasets, transforms
from utils.sampling import noniid
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#########################
#### Dataset loading ####
#########################
def read_femnist_dataset(dataset="femnist", 
                         num_clients = 100, 
                         trainer = "fedeq_admm", 
                         data_dir='data/femnist', format=".pkl", **__kwargs):

    train_file = dataset + '_train' + format
    test_file = dataset + '_test' + format


    trainObj = open(os.path.join(data_dir, train_file), 'rb')
    mat_train = pickle.load(trainObj)
    trainObj.close()

    testObj = open(os.path.join(data_dir, test_file), 'rb')
    mat_test = pickle.load(testObj)
    testObj.close()

    x_trains = mat_train['x'][:num_clients]
    y_trains = mat_train['y'][:num_clients]
    x_tests = mat_test['x'][:num_clients]
    y_tests = mat_test['y'][:num_clients]

    assert len(x_trains) == len(y_trains) == len(x_tests) == len(y_tests)
    num_clients = len(x_trains)

    y_trains = [y_trains[i].reshape(-1) for i in range(num_clients)]
    y_tests = [y_tests[i].reshape(-1) for i in range(num_clients)]

    list_train_samples = np.array([len(x_trains[i]) for i in range(num_clients)])
    num_train = np.sum(list_train_samples)

    # Calculate mean, std
    mean = np.mean(list_train_samples)
    std = np.sqrt(np.var(list_train_samples))

    list_test_samples = np.array([len(x_tests[i]) for i in range(num_clients)])
    num_test = np.sum(list_test_samples)

    print(f'{dataset} dataset:')
    print('\tnumber of clients:', num_clients)
    print('\ttrain examples:', list_train_samples)
    print('\tnumber of train examples:', num_train)
    print('\tGlobal mean:', mean)
    print('\tGlobal std:', std)
    print('\ttest examples:', list_test_samples)
    print('\tnumber of train examples:', num_test)
    print('\tnumber of train labels:', [len(np.unique(y_trains[i])) for i in range(num_clients)])
    print('\tnumber of test labels:', [len(np.unique(y_tests[i])) for i in range(num_clients)])
    print('\tnumber of features:', x_trains[0][0].shape)
    print('\tshape of xtrains:', x_trains[0].shape)
    print('\tshape of ytrains:', y_trains[0].shape)
    print('\tshape of xtests:', x_tests[0].shape)
    print('\tshape of ytests:', y_tests[0].shape)

    return x_trains, y_trains, x_trains, y_trains, x_tests, y_tests
    

def read_cifar10_data(num_clients = 100, 
                      trainer = "fedeq_admm", 
                      num_labels = 5, 
                      data_dir='data/cifar10', format=".pkl", **__kwargs):
  """Read CIFAR-10 data."""

  train_file = 'cifar10_train_100' +  "_" + str(num_labels) + format
  test_file = 'cifar10_test_100' +  "_" + str(num_labels) + format

  trainObj = open(os.path.join(data_dir, train_file), 'rb')
  mat_train = pickle.load(trainObj)
  trainObj.close()

  testObj = open(os.path.join(data_dir, test_file), 'rb')
  mat_test = pickle.load(testObj)
  testObj.close()

  x_trains = mat_train['x'][:num_clients]
  y_trains = mat_train['y'][:num_clients]
  x_tests = mat_test['x'][:num_clients]
  y_tests = mat_test['y'][:num_clients]

  assert len(x_trains) == len(y_trains) == len(x_tests) == len(y_tests)
  num_clients = len(x_trains)
  y_trains = [y_trains[i].reshape(-1) for i in range(num_clients)]
  y_tests = [y_tests[i].reshape(-1) for i in range(num_clients)]
  print('CIFAR-10 dataset:')
  print('\tnumber of clients:', num_clients)
  print('\tnumber of train examples:', [len(x_trains[i]) for i in range(num_clients)])
  print('\tnumber of test examples:', [len(x_tests[i]) for i in range(num_clients)])
  print('\tnumber of train labels:', [len(np.unique(y_trains[i])) for i in range(num_clients)])
  print('\tnumber of test labels:', [len(np.unique(y_tests[i])) for i in range(num_clients)])
  print('\tnumber of features:', x_trains[0][0].shape)
  print('\tshape of xtrains:', x_trains[0].shape)
  print('\tshape of ytrains:', y_trains[0].shape)
  print('\tshape of xtests:', x_tests[0].shape)
  print('\tshape of ytests:', y_tests[0].shape)

  return x_trains, y_trains, x_trains, y_trains, x_tests, y_tests

def read_cifar100_data(num_clients = 100, 
                       trainer="fedeq_admm", 
                       num_labels = 5, 
                       data_dir='data/cifar100', format=".mat", **__kwargs):
  """Read CIFAR100 data."""

  train_file = 'cifar100_train_' + str(num_clients) + "_" + str(num_labels) + format
  test_file = 'cifar100_test_' + str(num_clients) + "_" + str(num_labels) + format
  mat_train = scipy.io.loadmat(os.path.join(data_dir, train_file))
  mat_test = scipy.io.loadmat(os.path.join(data_dir, test_file))


  if num_clients == 1:
    x_trains = mat_train['x']
    y_trains = mat_train['y']
    x_tests = mat_test['x']
    y_tests = mat_test['y']
  else:
    x_trains = mat_train['x'][:num_clients]
    y_trains = mat_train['y'][:num_clients]
    x_tests = mat_test['x'][:num_clients]
    y_tests = mat_test['y'][:num_clients]

  print(len(x_trains))
  assert len(x_trains) == len(y_trains) == len(x_tests) == len(y_tests)
  num_clients = len(x_trains)
  y_trains = [y_trains[i].reshape(-1) for i in range(num_clients)]
  y_tests = [y_tests[i].reshape(-1) for i in range(num_clients)]
  print('CIFAR-100 dataset:')
  print('\tnumber of clients:', num_clients)
  print('\tnumber of train examples:', [len(x_trains[i]) for i in range(num_clients)])
  print('\tnumber of test examples:', [len(x_tests[i]) for i in range(num_clients)])
  print('\tnumber of train labels:', [len(np.unique(y_trains[i])) for i in range(num_clients)])
  print('\tnumber of test labels:', [len(np.unique(y_tests[i])) for i in range(num_clients)])
  print('\tnumber of features:', x_trains[0][0].shape)
  print('\tshape of xtrains:', x_trains[0].shape)
  print('\tshape of ytrains:', y_trains[0].shape)
  print('\tshape of xtests:', x_tests[0].shape)
  print('\tshape of ytests:', y_tests[0].shape)

  return x_trains, y_trains, x_trains, y_trains, x_tests, y_tests

def read_shakespeare_dataset(dataset="shakespeare", 
                             num_clients = 100, 
                             trainer = "fedeq_admm", 
                             num_labels = 16, 
                             data_dir='data/shakespeare', format=".pkl", **__kwargs):
    """Read SHAKESPEARE data."""

    train_file = dataset + '_train' + "_" + str(num_labels) + format
    test_file = dataset + '_test' + "_" + str(num_labels) + format


    trainObj = open(os.path.join(data_dir, train_file), 'rb')
    mat_train = pickle.load(trainObj)
    trainObj.close()

    testObj = open(os.path.join(data_dir, test_file), 'rb')
    mat_test = pickle.load(testObj)
    testObj.close()

    x_trains = mat_train['x'][:num_clients]
    y_trains = mat_train['y'][:num_clients]
    x_tests = mat_test['x'][:num_clients]
    y_tests = mat_test['y'][:num_clients]

    assert len(x_trains) == len(y_trains) == len(x_tests) == len(y_tests)
    num_clients = len(x_trains)


    list_train_samples = np.array([len(x_trains[i]) for i in range(num_clients)])
    num_train = np.sum(list_train_samples)

    # Calculate mean, std
    mean = np.mean(list_train_samples)
    std = np.sqrt(np.var(list_train_samples))

    list_test_samples = np.array([len(x_tests[i]) for i in range(num_clients)])
    num_test = np.sum(list_test_samples)

    print(f'{dataset} dataset:')
    print('\tnumber of clients:', num_clients)
    print('\tlist of train examples:', list_train_samples)
    print('\tnumber of train examples:', num_train)
    print('\tlist of test examples:', list_test_samples)
    print('\tnumber of test examples:', num_test)
    print('\tmean:', mean)
    print('\tstd:', std)
    print('\tnumber of features:', x_trains[0][0].shape)
    print('\tshape of xtrains:', x_trains[0].shape)
    print('\tshape of ytrains:', y_trains[0].shape)
    print('\tshape of xtests:', x_tests[0].shape)
    print('\tshape of ytests:', y_tests[0].shape)

    return x_trains, y_trains, x_trains, y_trains, x_tests, y_tests


#########################
#### Dataset Utils ######
#########################

def gen_batch(data_x, data_y, batch_size, num_iter):
  """NOTE: Deprecated in favor of `epoch_generator`."""
  index = len(data_y)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data_y)):
      index = 0
      data_x, data_y = sklearn.utils.shuffle(data_x, data_y, random_state=i + 1)

    batched_x = data_x[index:index + batch_size]
    batched_y = data_y[index:index + batch_size]

    yield (batched_x, batched_y)


def epochs_generator(data_x, data_y, batch_size, epochs=1, seed=None):
  for ep in range(epochs):
    gen = epoch_generator(data_x, data_y, batch_size, seed=seed + ep)
    for batch in gen:
      yield batch


def epoch_generator(data_x, data_y, batch_size, seed=None):
  """Generate one epoch of batches."""
  data_x, data_y = sklearn.utils.shuffle(data_x, data_y, random_state=seed)
  # Drop last by default
  epoch_iters = len(data_x) // batch_size
  for i in range(epoch_iters):
    left, right = i * batch_size, (i + 1) * batch_size
    yield (data_x[left:right], data_y[left:right])


def client_selection(seed, total, num_selected, weights=None):
  rng = np.random.default_rng(seed=seed)
  indices = rng.choice(range(total), num_selected, replace=False, p=weights)
  return indices


def print_log(message, fpath=None, stdout=True, print_time=False):
  if print_time:
    timestr = time.strftime('%Y-%m-%d %a %H:%M:%S')
    message = f'{timestr} | {message}'
  if stdout:
    print(message)
  if fpath is not None:
    with open(fpath, 'a') as f:
      print(message, file=f)

def save_plot(message, fpath=None):

  iters = range(len(message[0]))
  sns.set_style("whitegrid")

  plt.plot(iters, message[0])
  plt.plot(iters, message[1])

  plt.xlabel('Iterations')
  plt.ylabel('Accuracy')

  plt.legend(['Train', 'Test'])

  plt.savefig(fpath)  
