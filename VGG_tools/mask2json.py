import argparse
import cv2
import json
import os
import numpy as np

'''
This script convert img files to json vgg data.
'''
# Input management
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask", type=str, required=False, default=r"C:\Users\Susi\Desktop\94_c0.png")
ap.add_argument("-o", "--output", type=str, required=False, default=r"C:\Users\Susi\Desktop\prueba.json")
ap.add_argument("-l", "--label", type=str, required=False, default="licence_plate")


def write_json(data, out):
    with open(out, 'w') as o:
        json.dump(data, o)
        o.close()


def load_data(label, img_path, w, h, points_x, points_y):
    data = {}
    region = {}
    index = str(img_path.split('\\')[-1])
    for i in range(len(points_x)):
        region[i] = {
            "shape_attributes": {
                "name": "polygon",
                "all_points_x":
                    points_x[i]
                ,
                "all_points_y":
                    points_y[i]

            },
            "region_attributes": {
                "label": label
            }
        }
    data[index] = {
        "fileref": "",
        "size": w * h,
        "filename": img_path.split('\\')[-1],
        "base64_img_data": "",
        "file_attributes": {},
        "regions": region
    }
    return data


def get_points(img):
    points_x = []
    points_y = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        aux_x = []
        aux_y = []
        for p in range(c.shape[0]):
            aux_x.append(int(c[p][0][0]))
            aux_y.append(int(c[p][0][1]))
        points_x.append(aux_x)
        points_y.append(aux_y)

    return points_x, points_y


if __name__ == "__main__":
    args = vars(ap.parse_args())
    img_path = args['mask']
    output_path = args['output']
    label = args['label']

    img = cv2.imread(img_path)

    points_x, points_y = get_points(img)
    data = load_data(label, img_path, img.shape[1], img.shape[0], points_x, points_y)
    write_json(data, output_path)
