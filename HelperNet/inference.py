from HelperNet.Tools.NetManager import *
import sys

'''
This script executes the network inference.
'''

if __name__ == '__main__':
    # Variables
    path_images = r"C:\Users\TTe_J\Downloads\RGB"
    path_images_destination = r"C:\Users\TTe_J\Downloads\new_RGBs"
    path_json = r'C:\Users\TTe_J\Downloads\labels_vgg.json'
    path_newlabels = r'C:\Users\TTe_J\Downloads\new_labels.json'
    limit = 40 # unlabeled image limit
    model = "HelperNetV1" # models = HelperNetV1
    start_epoch = 232 # <= trained epochs
    input_dims = (720, 1280, 3)
    weights_path = f"./Models/{model}/epoch_{start_epoch}"
    labels = ['license_plate', 'head']

    # Overwrite control
    if os.path.exists(path_newlabels):
        over = input(f"WARNING!!! Existing labels will be overwritten (overwrite or stop: o/s) => ")
        if over != 'o':
            print("Stoping")
            sys.exit()
    if os.path.exists(path_images_destination):
        shutil.rmtree(path_images_destination)
    os.makedirs(path_images_destination)

    # Cargo el modelo y sus pesos
    mod = load_mod(model, dim=input_dims)
    mod.load_weights(weights_path)

    # Load unlabeled images
    unlabed, names, sizes = loadUnlabeled(path_json, path_images, path_images_destination, limit)

    # Predict de una imagen en concreto
    y_hat = mod.predict(unlabed)

    # Save as json and show names of the modified files
    vgg = mask2vgg(np.round(y_hat).astype(np.uint8), labels, names, sizes, save_path=path_newlabels)
    print(names)
