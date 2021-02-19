import cv2, json, os
import numpy as np
import matplotlib.pyplot as plt

def mask2vgg(masks, labels, names, sizes, save_path=None):
    file = {}
    for i in range(len(masks)):
        regions = {}
        counter = 0
        for m in range(masks[i].shape[2]-1):
            contours, _ = cv2.findContours(masks[i][:, :, m], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in range(len(contours)):
                regions[str(counter)] = {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": contours[c][:, :, 0],
                        "all_points_y": contours[c][:, :, 1]
                    },
                    "region_attributes": {
                        "label": labels[m]
                    }
                }
                counter += 1
        file[names[i]] = {
            "fileref": "",
            "size": sizes[i],
            "filename": names[i],
            "base64_img_data": "",
            "file_attributes": {},
            "regions": regions
        }

    if save_path != None:
        json.dump(file, save_path)
    return file

if __name__ == '__main__':
    labels=['license_plate', 'head']
    path = r'C:\Users\TTe_J\Downloads\testmask.png'
    out = r'C:\Users\TTe_J\Downloads\prueba.json'
    names = ["0asdf", "1kjnd", "2asdf"]
    sizes = [123123, 23123, 32123]

    a = np.array([[1], [3], [5]])
    b = np.array([[2], [4], [6]])
    a.size

    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a[:,0]
    c[1::2] = b[:,0]

    print(a.size)