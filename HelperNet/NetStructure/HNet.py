from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.regularizers import L2

'''
This script contains the proposed network structures.
'''

def HelperNetV1(inputs, learn_reg=1e-2):
  # x.shape = 1280x720x3
  filters=[2, 4, 8, 64, 128, 3]
  kernel_sizes = [(5, 5), (3, 3), (1, 1)]
  downup = (2, 2)
  activation = "relu"
  lreg = L2(learn_reg)

  # Lado izquierdo = Li, Medio = M y Lado derecho = Ld
  # - Nivel 1, Li
  # n1Li = Dropout(0.1)(inputs)
  n1Li = Conv2D(filters=filters[0], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(inputs) # 1280x720x2
  n1Li = BatchNormalization()(n1Li)

  # - Nivel 2, Li
  n2Li = MaxPooling2D(downup)(n1Li) # 640x360x2
  n2Li = Conv2D(filters=filters[1], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n2Li) # 640x360x4
  n2Li = BatchNormalization()(n2Li)

  # - Nivel 3, Li
  n3Li = MaxPooling2D(downup)(n2Li) # 320x180x4
  n3Li = Conv2D(filters=filters[2], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n3Li) # 320x180x8
  n3Li = BatchNormalization()(n3Li)

  # - Nivel 4, Li
  n4Li = MaxPooling2D(downup)(n3Li) # 180x90x8
  n4Li = Conv2D(filters=filters[3], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n4Li) # 180x90x16
  n4Li = BatchNormalization()(n4Li)

  # - Nivel 5, M
  n5M = Conv2D(filters=filters[4], kernel_size=kernel_sizes[0], kernel_regularizer=lreg, padding="same", activation=activation)(n4Li) # 180x90x32
  n5M = BatchNormalization()(n5M)

  # - Nivel 4, Ld
  n4Ld = concatenate([n4Li, n5M], axis=3) # 180x90x48
  n4Ld = Conv2D(filters=filters[3], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n4Ld) # 180x90x16
  n4Ld = UpSampling2D(downup)(n4Ld) # 320x180x16
  n4Ld = BatchNormalization()(n4Ld)

  # - Nivel 3, Ld
  n3Ld = concatenate([n3Li, n4Ld]) #320x180x24
  n3Ld = Conv2D(filters=filters[2], kernel_size=kernel_sizes[2], kernel_regularizer=lreg, padding="same", activation=activation)(n3Ld) # 320x180x8
  n3Ld = UpSampling2D(downup)(n3Ld) # 640x360x8
  n3Ld = BatchNormalization()(n3Ld)

  # - Nivel 2, Ld
  n2Ld = concatenate([n2Li, n3Ld]) #640x360x12
  n2Ld = Conv2D(filters=filters[1], kernel_size=kernel_sizes[2], kernel_regularizer=lreg, padding="same", activation=activation)(n2Ld) # 640x360x4
  n2Ld = UpSampling2D(downup)(n2Ld) # 1280x720x4
  n2Ld = BatchNormalization()(n2Ld)

  # - Nivel 1, Ld
  n1Ld = concatenate([n1Li, n2Ld]) # 1280x720x6
  n1Ld = Conv2D(filters=filters[5], kernel_size=kernel_sizes[2], kernel_regularizer=lreg, padding="same", activation="softmax")(n1Ld) # 1280x720x2
  return n1Ld