import json
import argparse
'''
This script change the name of images on VGG.json.
'''

# Input management
ap = argparse.ArgumentParser()
ap.add_argument("-fj", "--file_json", type=str, required=False, default=r"s0_4_coco.json")
ap.add_argument("-o", "--output", type=str, required=False, default=r"C:\Users\Susi\Desktop\prueba.json")
ap.add_argument("-t", "--tag", type=str, required=False, default="s0_0_")


if __name__ == "__main__":
    args = vars(ap.parse_args())
    json_path = args['file_json']
    output_path = args['output']
    tag = args['tag']

    with open(json_path) as i:
        data = json.load(i)
        i.close()
    for image in data['images']:
        image['file_name'] = tag + image['file_name']

    with open(output_path, "w") as o:
        json.dump(data, o)
        o.close()