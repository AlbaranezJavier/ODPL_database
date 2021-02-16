import json
import argparse
'''
This script change the name of images on VGG.json.
'''

# Input management
ap = argparse.ArgumentParser()
ap.add_argument("-fj", "--file_json", type=str, required=False, default=r"C:\Users\Susi\Desktop\mini-dataset-aparcamiento\labels_s3_0.json")
ap.add_argument("-o", "--output", type=str, required=False, default=r"C:\Users\Susi\Desktop\nuevas_etiquetas\s3_0_vgg.json")
ap.add_argument("-t", "--tag", type=str, required=False, default="s3_0_")


if __name__ == "__main__":
    args = vars(ap.parse_args())
    json_path = args['file_json']
    output_path = args['output']
    tag = args['tag']

    with open(json_path) as i:
        data = json.load(i)
        i.close()
    keys = list(data.keys())
    for k in range(len(keys)):
        data[keys[k]]['filename'] = tag+keys[k]
        data[tag+keys[k]] = data.pop(keys[k])

    with open(output_path, "w") as o:
        json.dump(data, o)
        o.close()