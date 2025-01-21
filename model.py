import keras
import tensorflow.keras.backend as K
import cv2
import numpy as np
import string

# Define Model
char_list = string.ascii_letters+string.digits

def CRNN_Model():
  inputs = keras.layers.Input(shape=(32,128,1))
  x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv_1')(inputs)
  x = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, name='pool_1')(x)
  x = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv_2')(x)
  x = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, name='pool_2')(x)
  x = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', name='conv_3')(x)
  x = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', name='conv_4')(x)
  # Rectangular pool window 2x1 to get wider feature map and shrink feature map accross height.
  x = keras.layers.MaxPooling2D(pool_size=(2,1), name='pool_3')(x)
  x = keras.layers.Conv2D(512, (3,3), activation='relu', padding='same', name='conv_5')(x)
  x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
  x = keras.layers.Conv2D(512, (3,3), activation='relu', padding='same', name='conv_6')(x)
  x = keras.layers.BatchNormalization(name='batch_norm_2')(x)
  x = keras.layers.MaxPooling2D(pool_size=(2,1), name='pool_4')(x)
  x = keras.layers.Conv2D(512, (2,2), activation='relu')(x)

  # LSTM Block
  # Input: (31, 512)
  x = keras.layers.Lambda(lambda x: K.squeeze(x, 1), name='squeeze_input')(x)
  x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.2), name='BiLSTM_1')(x)
  x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.2), name='BiLSTM_2')(x)
  dense_output = keras.layers.Dense(len(char_list)+1, activation='softmax')(x)

  return keras.models.Model(inputs=[inputs], outputs=dense_output)

# Load weights
model = CRNN_Model()
model.load_weights('./models/best_model.keras')

def get_prediction(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128,32))
    image = np.expand_dims(image , axis = 2)
    image = image/255.
    image = image.reshape(1, 32, 128, 1)
    prediction = model.predict(image)
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1], greedy=True)[0][0])
    pred = ""
    for p in out[0]:
        if int(p) != -1:
            pred += char_list[int(p)]
    return pred