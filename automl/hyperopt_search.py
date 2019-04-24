import matplotlib as plt
import numpy as np
import argparse
import os
import time
import math

# sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics
from skimage import io
from sklearn.externals import joblib
from sklearn import preprocessing

# hyperopt
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import Trials

# feature extraction
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
import cv2
from scipy.cluster.vq import *

HEIGHT = 512
WIDTH = 512
train_path = None
val_path = None
x_train = None
y_train = None
x_val = None
y_val = None

name = ""
run_no = 0

def get_dataset_size(dataset_path):
  size = 0
  for directory in os.listdir(dataset_path):
    path = os.path.join(dataset_path, directory)
    size += len(os.listdir(dataset_path + directory))
  return size


def load_data(train_path, val_path, im_size=HEIGHT*WIDTH):
  train_size = get_dataset_size(train_path)
  val_size = get_dataset_size(val_path)
  print("Found {} images for training and {} for validation".format(train_size, val_size))
  x_train = np.zeros((train_size, im_size))
  y_train = np.zeros(train_size)
  x_val = np.zeros((val_size, im_size))
  y_val = np.zeros(val_size)


  print("Begin loading training dataset")
  i = 0
  for directory in sorted(os.listdir(train_path)):
    path = os.path.join(train_path, directory)
    print("{}% done".format((i/train_size)*100))
    for filename in sorted(os.listdir(path)):
      full_path = os.path.join(path, filename)
      img = io.imread(full_path, as_gray=True)
      x_train[i] = img.flatten()
      if (len(os.listdir(val_path)) == 2):
        y_train[i] = 1 if (int(directory) <= 2) else 2
      else:
        y_train[i] = directory
      i += 1


  print("Begin loading validation dataset", time.time())
  t0 = time.time()
  i = 0
  for directory in sorted(os.listdir(val_path)):
    path = os.path.join(val_path, directory)
    print("{}% done".format((i/val_size)*100))
    for filename in sorted(os.listdir(path)):
      full_path = os.path.join(path, filename)
      img = io.imread(full_path, as_gray=True)
      x_val[i] = img.flatten()
      y_val[i] = directory
      i += 1

  return x_train, y_train, x_val, y_val




def extract_LBP(images, radius=3, points=8, method='uniform', verbose=1): 
  n_points = points
  features = []
  for i, image in enumerate(images):
    if ((verbose) and (i % int(0.1*images.shape[0]) == 0)):
        print("Image", i, "of", images.shape[0])
    lbp = local_binary_pattern(image.reshape((HEIGHT, WIDTH)), n_points, radius, method)
    n_bins = int(lbp.max() + 1)
    feature, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    features.append(feature)

  features = np.array(features)
  return features


def extract_GLCM(images, distances=[5], angles=[0], symmetric=True, normed=True, verbose=1):
  features = []
  for i, image in enumerate(images):
    if ((verbose) and (i % int(0.1*images.shape[0]) == 0)):
        print("Image", i, "of", images.shape[0])
    glcm = greycomatrix(image.reshape((HEIGHT, WIDTH)).astype(np.uint8), distances, angles, symmetric=symmetric, normed=normed)
    im_features = []
    for metric in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'):
      for feature_list in greycoprops(glcm, metric):
          for feature in feature_list:
              im_features.append(feature)

    features.append(im_features)

  features = np.array(features)
  return features


def extract_SIFT(path, k=100): 
    sift = cv2.xfeatures2d.SIFT_create()
    des_list = []
    labels = []
    i = 0
    for directory in os.listdir(path):
        full_path = os.path.join(path, directory)
        for filename in os.listdir(full_path):
            img_path = os.path.join(full_path, filename)
            img = io.imread(img_path, as_gray=True)
            kp, des = sift.detectAndCompute(img, None)
            des_list.append((i, des))
            if (len(os.listdir(val_path)) == 2):
              label = 1 if (int(directory) <= 2) else 2
              labels.append(label)
            else:
              labels.append(directory)
            i += 1

    print("Created descriptors", time.time())
    descriptors = des_list[0][1]
    for i, descriptor in des_list[1:]:
        if descriptor is not None:
            descriptors = np.vstack((descriptors, descriptor))
    im_features = np.zeros((len(des_list), k), np.float32)
    if descriptors is None:
        return im_features, np.array(labels)
    voc, variance = kmeans(descriptors, k, 1)

    print("Computed kmeans", time.time())
    for i in range(len(des_list)):
        if des_list[i][1] is not None:
            words, distance = vq(des_list[i][1], voc)
            for w in words:
                im_features[i][w] += 1
    return im_features, np.array(labels) 


# Adjust weights to binary format if necessary
def adjust_class_weight(class_weight):
  number_of_labels = len(os.listdir(val_path))
  if (type(class_weight) == dict) and (number_of_labels == 2):
    class_weight[1] = class_weight[1] + class_weight[2]
    class_weight[2] = class_weight[3] + class_weight[4] + class_weight[5]
    class_weight.pop(3, None)
    class_weight.pop(4, None)
    class_weight.pop(5, None)
  return class_weight


def full_objective(space):
    global run_no
    print("Beginning evaluation for run:", run_no, time.time())
    print(space)

    extractor = space['descriptor']['name']
    if (extractor == 'lbp'):
        R = space['descriptor']['R']
        P = space['descriptor']['P']
        _x_train = extract_LBP(x_train, radius=R, points=P, method='uniform', verbose=0)
        _x_val = extract_LBP(x_val, radius=R, points=P, method='uniform', verbose=0)
        scaler = preprocessing.StandardScaler().fit(_x_train)
        _x_train = scaler.transform(_x_train)
        _x_val = scaler.transform(_x_val)
    elif (extractor == 'glcm'):
        distances = space['descriptor']['distances']
        angles = space['descriptor']['angles']
        _x_train = extract_GLCM(x_train, distances=distances, angles=angles, symmetric=True, normed=True, verbose=0)
        _x_val = extract_GLCM(x_val, distances=distances, angles=angles, symmetric=True, normed=True, verbose=0)
        scaler = preprocessing.StandardScaler().fit(_x_train)
        _x_train = scaler.transform(_x_train)
        _x_val = scaler.transform(_x_val)
    elif (extractor == 'sift'):
        k = space['descriptor']['k']
        _x_train, _ = extract_SIFT(train_path, k=k)
        _x_val, _ = extract_SIFT(val_path, k=k)
        scaler = preprocessing.StandardScaler().fit(_x_train)
        _x_train = scaler.transform(_x_train)
        _x_val = scaler.transform(_x_val)

    model = space['classifier']['name']
    if (model == 'rf'):
        n_estimators = space['classifier']['n_estimators']
        max_features = space['classifier']['max_features']
        max_depth    = space['classifier']['max_depth']
        class_weight = space['classifier']['class_weight']
        class_weight = adjust_class_weight(class_weight)
        classifier = RFC(
                n_estimators=n_estimators,
                max_features=max_features,
                max_depth=max_depth,
                class_weight=class_weight
                )
    elif (model == 'svm'):
        C            = space['classifier']['C']
        gamma        = space['classifier']['gamma']
        class_weight = space['classifier']['class_weight']
        class_weight = adjust_class_weight(class_weight)
        classifier = svm.SVC(
                C = C,
                gamma = gamma,
                class_weight = class_weight
                )
    else:
        print("unrecognized classifier:", model)
        return
    classifier.fit(_x_train, y_train)

    y_pred = classifier.predict(_x_val)
    acc = ((y_pred == y_val).sum())/y_val.shape[0]
    y_train_pred = classifier.predict(_x_train)
    train_acc = ((y_train_pred == y_train).sum())/y_train.shape[0]

    print("Confusion matrix train:\n%s" % metrics.confusion_matrix(y_train, y_train_pred))
    print("Confusion matrix val:\n%s" % metrics.confusion_matrix(y_val, y_pred))



    os.makedirs("hpfull_models", exist_ok=True)
    model_name = 'run' + str(run_no) + "-model"
    joblib.dump(classifier, "hpfull_models/" + model_name)

    print("Finished evaluation of run:", run_no, time.time())
    print("Run", run_no, "results", train_acc, acc)
    run_no += 1

    return {
            'loss': (1 - acc),
            'status': STATUS_OK,
            'space': space,
            'val_acc': acc,
            'train_acc': train_acc,
            'eval_time': time.time()
            }


def main(objective, space, max_evals=150): 
    trials = Trials()
    best = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
            )

    for result in trials.results:
        print(result)
    print("Best configuration:", best)


def get_hyperopt_space():
    pi = math.pi
    space = {'classifier': hp.choice('classifier',[
                    {
                    'name': 'svm',
                    'C': hp.choice('C', [2**-5, 2**-3, 2**-1, 1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]),
                    'gamma': hp.choice('gamma', ["auto", 2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]),
                    'class_weight': hp.choice('class_weight_svm', 
                    [
                        None,
                        'balanced',
                        {
                            1: 0.35,
                            2: 0.35,
                            3: 0.100,
                            4: 0.100,
                            5: 0.100
                        },
                        {
                            1: 0.30,
                            2: 0.30,
                            3: 0.133,
                            4: 0.133,
                            5: 0.133
                        },
                        {
                            1: 0.25,
                            2: 0.25,
                            3: 0.166,
                            4: 0.166,
                            5: 0.166
                        }
                    ]) 
                    },
                    {
                    'name': 'rf',
                    'n_estimators': hp.choice('n_estimators', [10, 100, 1000]),
                    'max_features': hp.choice('max_features', ["auto", 0.25, 0.5, 0.75]),
                    'max_depth'   : hp.choice('max_depth', [None, 2, 4, 6, 8, 10, 20]),
                    'class_weight': hp.choice('class_weight_rf', [
                        None,
                        'balanced',
                        {
                            1: 0.35,
                            2: 0.35,
                            3: 0.100,
                            4: 0.100,
                            5: 0.100
                        },
                        {
                            1: 0.30,
                            2: 0.30,
                            3: 0.133,
                            4: 0.133,
                            5: 0.133
                        },
                        {
                            1: 0.25,
                            2: 0.25,
                            3: 0.166,
                            4: 0.166,
                            5: 0.166
                        }
                        ]),
                    }
            ]),
            'descriptor': hp.choice('descriptor',[
                {
                    'name': 'glcm',
                    'distances': hp.choice('distances', [
                                                        [1], [2], [4], [5], [8],
                                                        [1, 2], [1, 4], [1, 5], [1, 8], [2, 4], [2, 5], [2, 8], [4, 5], [4, 8], [5, 8],
                                                        [1, 2, 4], [1, 2, 5], [1, 2, 8], [1, 4, 5], [1, 4, 8], [1, 5, 8], [2, 4, 5], [2, 4, 8], [2, 5, 8], [4, 5, 8],
                                                        [1, 2, 4, 5], [1, 2, 4, 8], [1, 2, 5, 8], [1, 4, 5, 8], [2, 4, 5, 8],
                                                        [1, 2, 4, 5, 8]
                                                        ]),
                    'angles': hp.choice('angles', [
                                                    [0], [pi/4], [pi/2], [3*pi/4],
                                                    [0, pi/4], [0, pi/2], [0, 3*pi/4], [pi/4, pi/2], [pi/4, 3*pi/4], [pi/2, 3*pi/4],
                                                    [0, pi/4, pi/2], [0, pi/4, 3*pi/4], [0, pi/2, 3*pi/4], [pi/4, pi/2, 3*pi/4],
                                                    [0, pi/4, pi/2, 3*pi/4]
                                                  ]),
                },
                {
                    'name': 'lbp',
                    'P': hp.choice('P', [4, 8, 16, 24]),
                    'R': hp.choice('R', [1, 2, 3]),
                },
                {
                    'name': 'sift',
                    'k': hp.choice('k', [10, 25, 50, 100, 250, 500, 1000, 5000])
                }
                ])
            }
    return space

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Hyperopt on DiffraNet')
  parser.add_argument('--train_path', dest='train_path', action='store', default='../data/synthetic/train/')
  parser.add_argument('--val_path', dest='val_path', action='store', default='../data/real_preprocessed/validation/')
  parser.add_argument('--iters', dest="iters", type=int, action='store', default=150)
  args = parser.parse_args()

  train_path = args.train_path
  val_path = args.val_path
  x_train, y_train, x_val, y_val = load_data(args.train_path, args.val_path)
  space = get_hyperopt_space()
  main(full_objective, space, args.iters)

  print("The entire script took:", time.time() - start_time, "seconds to run")
