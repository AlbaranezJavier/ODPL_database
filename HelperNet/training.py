from sklearn.model_selection import train_test_split
import time
from datetime import datetime
from HelperNet.Tools.NetManager import *

'''
This script executes the training of the network.
'''

if __name__ == '__main__':
    # Data preparation
    inputs_rgb = r'C:\Users\TTe_J\Downloads\RGB'
    inputs_json = r'C:\Users\TTe_J\Downloads\labels_vgg.json'
    labels = ["license_plate", "head"]
    seed = 123
    label_size = (720, 1280, 3)
    split = 0.2
    batch_size = 16
    test_size = 0
    valid_size = .10
    shuffle = True

    rgb_paths, gt_paths = getpaths(inputs_json, inputs_rgb, labels)
    X_train, X_valid, Y_train, Y_valid = train_test_split(rgb_paths, gt_paths, test_size=valid_size, random_state=seed,
                                                          shuffle=shuffle)

    print(f'Size: {len(rgb_paths)}')
    print(f'Train: {len(X_train)} y valid: {len(X_valid)}')

    # Mini-batches
    batches_valid = batch_division(X_valid, batch_size)
    batches_train = batch_division(X_train, batch_size)

    print(
        f'Train batches: {len(batches_train)}, valid batches: {len(batches_valid)}')

    # Variables
    model = "HelperNetV1"  # models = HelperNetV1
    start_epoch = 0 # <= numero de epocas que ya ha entrenado
    end_epoch = 250
    learn_opt, learn_reg = 1e-3, 1e-2
    save_weights = True
    weights_path = f'./Models/{model}/epoch_'
    weights_path2load = f'{weights_path}{start_epoch}'
    input_dims = (720, 1280, 3)

    # Model selection
    mod, optimizer, loss_fn, train_acc_metric, valid_acc_metric, logdir = load4training(model, dim=input_dims,
                                                                                   learn_opt=learn_opt, learn_reg=learn_reg,
                                                                                   start_epoch=start_epoch)

    print(f'Modelo {model}_epoch{start_epoch} creado!')
    if start_epoch > 0:
        mod.load_weights(weights_path2load)
        print(f'Pesos del modelo {model}_epoch{start_epoch} cargados con éxito!')

    # metrics management
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    now = datetime.now()
    writer = open(logdir + now.strftime("%d_%m_%Y__%H_%M_%S") + ".txt", "a")
    writer.write(f'epoch;loss_train;acc_train;acc_valid;min\n')
    writer.close()

    best = 0

    for epoch in range(start_epoch + 1, end_epoch + 1):
        # TRAINING
        start_time = time.time()
        train_loss, train_acc = 0, 0
        loss_value = 0
        for idx in range(len(batches_train) - 1):
            data = batch_x(X_train, batches_train[idx], batches_train[idx + 1])
            labels = batch_y(Y_train, batches_train[idx], batches_train[idx + 1], label_size)
            loss_value += train_step(data, labels, mod, loss_fn, optimizer, train_acc_metric)
            print('\r', f'Train_batch: {idx + 1}/{len(batches_train) - 1}', end='')
        train_loss, train_acc = loss_value / (len(batches_train) - 1), train_acc_metric.result() * 100.
        train_acc_metric.reset_states()

        # VALID
        for idx in range(len(batches_valid) - 1):
            data = batch_x(X_valid, batches_valid[idx], batches_valid[idx + 1])
            labels = batch_y(Y_valid, batches_valid[idx], batches_valid[idx + 1], label_size)
            valid_step(data, labels, mod, valid_acc_metric)
            print('\r', f'Valid_batch: {idx + 1}/{len(batches_valid)}         ', end='')
        valid_acc = valid_acc_metric.result() * 100.
        valid_acc_metric.reset_states()

        # Print and save the metrics
        end_time = round((time.time() - start_time) / 60, 2)
        writer = open(logdir + now.strftime("%d_%m_%Y__%H_%M_%S") + ".txt", "a")
        writer.write(f'{epoch};{train_loss};{train_acc};{valid_acc};{end_time}\n')
        writer.close()
        print('\r',
              f'Epoch {epoch}, Train_loss: {train_loss}, Train_acc: {train_acc}, Valid_acc: {valid_acc}, Time: {end_time}',
              end='')

        # Saves the weights of the model if it obtains the best result in validation
        if valid_acc > best and save_weights:
            best = valid_acc
            mod.save_weights(f'{weights_path}{epoch}')
            print(f' <= saved', end='')
        print()

