import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import os, json, cv2, shutil
from HelperNet.NetStructure.HNet import HelperNetV1

'''
This script contains all the necessary methods for training and inference processes.
'''

def load4training(model, dim, learn_opt, learn_reg, start_epoch):
  inputs = Input(shape=dim)
  if model == "HelperNetV1":
    logdir = f'./Logs/{model}_{start_epoch}/'
    mod = Model(inputs, HelperNetV1(inputs, learn_reg))
  else:
    print("ERROR load_mod")
  optimizer = RMSprop(learn_opt)
  loss_fn = CategoricalCrossentropy(from_logits=False)
  train_acc_metric = CategoricalAccuracy()
  valid_acc_metric = CategoricalAccuracy()
  mod.summary()
  return mod, optimizer, loss_fn, train_acc_metric, valid_acc_metric, logdir

def load4inference(model,dim):
  inputs = Input(shape=dim)
  if model == "HelperNetV1":
    mod = Model(inputs, HelperNetV1(inputs))
  else:
    print("ERROR load_mod")

  mod.summary()
  return mod

@tf.function
def train_step(x, y, model, loss_fn, optimizer, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def valid_step(x, y, model, valid_acc_metric):
    val_logits = model(x, training=False)
    valid_acc_metric.update_state(y, val_logits)

def getpaths(json_path, img_path, labels):
    REG, SATT, RATT, ALLX, ALLY, LAB, NAME = "regions", "shape_attributes", "region_attributes", "all_points_x", "all_points_y", "label", "filename"

    with open(json_path) as i:
        data = json.load(i)
        i.close()

    directories = []
    annotations = []

    for key in data.keys(): # each image
        path = data[key][NAME].split('-')
        directories.append(img_path + '/' + path[0] + '/' + data[key][NAME])
        regions = []
        if len(data[key][REG]) > 0: # could be empty
            regions = [[]]*len(labels)
            for i in range(len(data[key][REG])): # each region
                points = np.stack([data[key][REG][i][SATT][ALLX],data[key][REG][i][SATT][ALLY]], axis=1)
                for l in range(len(labels)): # depending label
                    if data[key][REG][i][RATT][LAB] == labels[l]:
                        regions[l].append(points)
                        break
        annotations.append(regions)
    return np.array(directories), np.array(annotations)

# Carga las imagenes entre idx y idx + 1
def batch_x(paths, idx, ending):
  x = [cv.imread(path).astype('float32') for path in paths[idx:ending]]
  return np.array(x) / 255.

def batch_y(labels, idx, ending, label_size):
  y = [get_mask(label, label_size) for label in labels[idx:ending]]
  return np.array(y)

# Divide el conjunto de entrenamiento en lotes
def batch_division(set, batch_size):
  batch_idx = np.arange(0,len(set), batch_size)
  return batch_idx

def get_mask(data, label_size):
    img = np.zeros(label_size, dtype=np.uint8)
    for lab in range(label_size[2]-1):
        zeros = np.zeros(label_size[0:2], dtype=np.uint8)
        for idx in range(len(data[lab])):
            cv2.fillConvexPoly(zeros, data[lab][idx], 1)
        img[:,:,lab] = zeros
    img[:,:,2] = np.logical_not(np.logical_or(img[:,:,0], img[:,:,1]))
    return img

def mask2vgg(masks, labels, names, sizes, save_path=None):
    file = {}
    for i in range(len(masks)):
        regions = []
        counter = 0
        for m in range(masks[i].shape[2]-1):
            contours, _ = cv2.findContours(masks[i][:, :, m], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in range(len(contours)):
                # At least three points to form a polygon
                countourX = []
                countourY = []
                if len(contours[c][:, :, 0]) > 2:
                    countourX = contours[c][:, :, 0][:, 0].tolist()
                    countourY = contours[c][:, :, 1][:, 0].tolist()

                regions.append({
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": countourX,
                        "all_points_y": countourY
                    },
                    "region_attributes": {
                        "label": labels[m]
                    }
                })
                counter += 1
        file[names[i]] = {

            "filename": names[i],
            "size": sizes[i],
            "regions": regions,
            "file_attributes": {}
        }

    if save_path != None:
        json_file = json.dumps(file, separators=(',', ':'))
        with open(save_path, "w") as outfile:
            outfile.write(json_file)
            outfile.close()
    return file

def mask2coco(masks, labels, names, save_path=None):
    categories = []
    images = []
    annotations = []
    for i in range(len(labels)):
        categories.append({
            "id": i,
            "name": labels[i]
        })

    counter = 0
    for i in range(len(masks)):
        images.append({
            "id": i+1,
            "width": masks.shape[2],
            "height": masks.shape[1],
            "file_name": names[i]
        })
        for m in range(masks[i].shape[2]-1):
            contours, _ = cv2.findContours(masks[i][:, :, m], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in range(len(contours)):
                points = []
                bbox = []
                # At least three points to form a polygon
                if len(contours[c][:, :, 0]) > 2:
                    points = np.empty((contours[c].size,), dtype=contours[c].dtype)
                    points[0::2] = contours[c][:, :, 0][:, 0]
                    points[1::2] = contours[c][:, :, 1][:, 0]

                    minx = contours[c][:, :, 0][:, 0].min()
                    maxx = contours[c][:, :, 0][:, 0].max()
                    miny = contours[c][:, :, 1][:, 0].min()
                    maxy = contours[c][:, :, 1][:, 0].max()

                    bbox = [minx, miny, maxx-minx, maxy-miny]
                annotations.append({
                    "id": counter,
                    "iscrowd": 0,
                    "image_id": i+1,
                    "category_id": m,
                    "segmentation": list(points),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3]
                })
                counter += 1
    file = {
        "info": {
            "description": "my-project_name"
        },
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    if save_path != None:
        json_file = json.dumps(file, separators=(',', ':'))
        with open(save_path, "w") as outfile:
            outfile.write(json_file)
            outfile.close()
    return file

def doNothing():
    return 0

def getInto(path, dest_path, labeled, unlabeled, names, sizes, limit, function=doNothing):
    stop = False
    [_, dirnames, filenames] = next(os.walk(path, topdown=True))
    for folder in dirnames:
        unlabeled, names, sizes, stop = getInto(path + "/" + folder, dest_path, labeled, unlabeled, names, sizes, limit, function=function)
        if stop:
            break
    for file in filenames:
        if len(unlabeled) >= limit:
            stop = True
            break
        unlabeled, names, sizes = function(path, dest_path, file, labeled, unlabeled, names, sizes)
    return unlabeled, names, sizes, stop

def isUnlabeled(path, dest_path, file, labeled, unlabeled, names, sizes):
    if file not in labeled:
        unlabeled.append(path + "/" + file)
        names.append(file)
        sizes.append(os.stat(unlabeled[-1]).st_size)
        shutil.copyfile(unlabeled[-1], dest_path + "/" + file)
    return unlabeled, names, sizes

def loadUnlabeled(path_json, path_images, path_images_dest, limit):
    with open(path_json) as i:
        data = json.load(i)
        i.close()

    labeled = set(data.keys())
    unlabeled = []
    names = []
    sizes = []

    unlabeled, names, sizes, _ = getInto(path_images, path_images_dest, labeled, unlabeled, names, sizes, limit, isUnlabeled)

    unlabeled = np.array([cv2.imread(unlab).astype('float32') for unlab in unlabeled]) / 255.

    return unlabeled, names, sizes