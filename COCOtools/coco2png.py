import cv2, json, argparse
import numpy as np

'''
This script combine coco.json and img files to png images.
'''

# Input management
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", type=str, required=False, default=r"C:\Users\TTe_J\Downloads\test")
ap.add_argument("-fj", "--file_json", type=str, required=False, default=r"\coco.json")
ap.add_argument("-fi", "--file_img", type=str, required=False, default=r"\susi.jpg")
ap.add_argument("-o", "--output", type=str, required=False, default=r"C:\Users\TTe_J\Downloads\test")
ap.add_argument("-iw", "--width", type=int, required=False, default=1175)
ap.add_argument("-ih", "--hight", type=int, required=False, default=780)
ap.add_argument("-v", "--verbose", type=int, required=False, default=1)


def coco2png(w, h, file_json, img, verbose=0):
    # get data from json
    with open(file_json) as f:
        data = json.load(f)

    # process data with coco notation
    points = np.array(data['annotations'][0]['segmentation'][0], np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.reshape(points, (points.size // 2, 2)), 255)

    # show the generated png if verbose
    if verbose == 1:
        temp = cv2.bitwise_not(mask) & img[:, :, 2]
        temp = cv2.merge((img[:, :, 0], img[:, :, 1], temp))
        cv2.imshow("prueba", temp)
        cv2.waitKey(0)

    return mask


if __name__ == "__main__":
    # imput manager
    args = vars(ap.parse_args())
    directory, file_json, file_img, output, width, hight, verbose = args['directory'], args['file_json'], args[
        'file_img'], args['output'], args['width'], args['hight'], args['verbose']

    # from json to png
    img = cv2.imread(directory+file_img)
    mask = coco2png(width, hight, directory + file_json, img, verbose)
    cv2.imwrite(output + r'\result.png', img)
