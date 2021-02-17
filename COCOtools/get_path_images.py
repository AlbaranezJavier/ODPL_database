import json
import argparse
import os
'''
This script get the path of images on VGG.json.
'''

# Input management
ap = argparse.ArgumentParser()
ap.add_argument("-fj", "--file_json", type=str, required=False, default=r"C:\Users\Susi\Desktop\mini-dataset-aparcamiento\labels_s3_0.json")
ap.add_argument("-i", "--img_path", type=str, required=False, default=r"C:\Users\Susi\Desktop\nuevas_etiquetas")

if __name__ == "__main__":
    args = vars(ap.parse_args())
    json_path = args['file_json']
    img_path = args['img_path']

    directories = []

    with open(json_path) as i:
        data = json.load(i)
        i.close()

    for image in data['images']:
        directory = image['file_name'].split('_')[0] + image['file_name'].split('_')[1]
        d = os.path.join(img_path, directory)
        directories.append(str(os.path.join(d, image['file_name'])))

    print(directories)