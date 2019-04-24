import matplotlib as plt
import numpy as np
import argparse
import os
import time
import math

# sklearn
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics
from skimage import io
from sklearn.externals import joblib
from sklearn import preprocessing

# feature extraction
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
import cv2
from scipy.cluster.vq import *

HEIGHT = 512
WIDTH = 512

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

  group_classes = (len(os.listdir(val_path)) == 2) and (len(os.listdir(train_path)) == 5) # True if only two classes on validation set, but 5 on the training set


  print("Begin loading training dataset")
  i = 0
  for directory in sorted(os.listdir(train_path)):
    path = os.path.join(train_path, directory)
    print("{}% done".format((i/train_size)*100))
    for filename in sorted(os.listdir(path)):
      full_path = os.path.join(path, filename)
      img = io.imread(full_path, as_gray=True)
      x_train[i] = img.flatten()
      if group_classes:
        y_train[i] = 1 if (int(directory) <= 2) else 2
      else:
        y_train[i] = directory
      i += 1


  print("Begin loading validation dataset", time.time())
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
  print("Finished loading val dataset", time.time())

  return x_train, y_train, x_val, y_val


def RF_classifier(x_train, y_train, x_val, y_val, max_features=0.75, n_estimators=100, max_depth=4, class_weight='balanced', name=""):

  classifier = RFC(max_depth=max_depth, max_features=max_features, n_estimators=n_estimators, class_weight=class_weight)

  print(time.time(), "Training Random Forest classifier")
  # train classifier
  classifier.fit(x_train, y_train)
  joblib.dump(classifier, name+'_RF_model.pkl')

  print(time.time(), "Validating classifier")
  # validate classifier
  y_pred_train = classifier.predict(x_train)
  y_pred_val = classifier.predict(x_val)

  train_acc = ((y_pred_train == y_train).sum())/y_train.shape[0]
  val_acc = ((y_pred_val == y_val).sum())/y_val.shape[0]

  # Results
  print("Train accuracy: {}, \nValidation accuracy: {}".format(train_acc*100, val_acc*100))

  #Training results
  print("Training classification report for RF:\n %s\n"% (metrics.classification_report(y_train, y_pred_train)))
  print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_train, y_pred_train))

  # Validation results
  print("Validation classification report for RF:\n %s\n"% (metrics.classification_report(y_val, y_pred_val)))
  print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_val, y_pred_val))


def extract_LBP(images, radius=3, points=8, method='uniform', verbose=1):
  n_points = points
  features = []
  for i, image in enumerate(images):
    if ((verbose) and (i % int(0.1*images.shape[0]) == 0)):
        print("Image", i, "of", images.shape[0])
    lbp = local_binary_pattern(image.reshape((512, 512)), n_points, radius, method)
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
    glcm = greycomatrix(image.reshape((512, 512)).astype(np.uint8), distances, angles, symmetric=symmetric, normed=normed)
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Classify images using RF')
  parser.add_argument('--train_path', dest='train_path', action='store', default='../data/synthetic/train/')
  parser.add_argument('--val_path', dest='val_path', action='store', default='../data/real_raw/validation/')
  parser.add_argument('--sift', action='store_true')
  parser.add_argument('--clusters', dest='clusters', type=int, action='store', default=100)
  parser.add_argument('--lbp', action='store_true')
  parser.add_argument('--radius', dest='radius', type=int, action='store', default=1)
  parser.add_argument('--points', dest='points', type=int, action='store', default=16)
  parser.add_argument('--glcm', action='store_true')
  parser.add_argument('--angles', dest='angles', nargs='+', action='store', default=[1,2,5,8])
  parser.add_argument('--distances', dest='distances', nargs='+', action='store', default=[math.pi/2, 3*math.pi/4])
  parser.add_argument('--max_depth', dest='max_depth', action='store', default=20)
  parser.add_argument('--max_features', dest='max_features', action='store', default=0.5)
  parser.add_argument('--n_estimators', dest='n_estimators', action='store', default=10)
  parser.add_argument('--class_weight', dest='class_weight', nargs='+', action='store', default=[0.35, 0.35, 0.1, 0.1, 0.1])
  parser.add_argument('--name', dest="name", action='store', default="")
  args = parser.parse_args()

  class_weight = args.class_weight
  if type(args.class_weight) == list:
    if (len(class_weight) == 5) and (len(os.listdir(args.val_path)) == 2):
      class_weight[0] = class_weight[0] + class_weight[1]
      class_weight[1] = class_weight[2] + class_weight[3] + class_weight[4]
    weights_dict = {}
    for i in range(len(os.listdir(args.val_path))):
      weights_dict[i+1] = class_weight[i]
    class_weight = weights_dict

  if(args.sift):
    x_train, y_train = extract_SIFT(args.train_path, k=clusters)
    x_val, y_val = extract_SIFT(args.val_path, k=clusters)
  elif(args.lbp):
    x_train, y_train, x_val, y_val = load_data(args.train_path, args.val_path)
    x_train = extract_LBP(x_train, radius=args.radius, points=args.points)
    x_val = extract_LBP(x_val, radius=args.radius, points=args.points)
  elif(args.glcm):
    x_train, y_train, x_val, y_val = load_data(args.train_path, args.val_path)
    x_train = extract_GLCM(x_train, distances=args.distances, angles=args.angles)
    x_val = extract_GLCM(x_val, distances=args.distances, angles=args.angles)
  else:
    x_train, y_train, x_val, y_val = load_data(args.train_path, args.val_path)

  scaler = preprocessing.StandardScaler().fit(x_train)
  x_train = scaler.transform(x_train)
  x_val = scaler.transform(x_val)

  print(time.time(), "Loaded dataset")
  RF_classifier(
      x_train,
      y_train,
      x_val,
      y_val,
      max_depth=args.max_depth,
      max_features=args.max_features,
      n_estimators=args.n_estimators,
      class_weight=class_weight,
      name=args.name
      )

