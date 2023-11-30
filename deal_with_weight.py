import keras
import numpy as np
from keras import layers
num_actions = 5

def create_q_model():
    global num_actions
    return keras.models.Sequential(
        [
            layers.Input(shape=(21, 21, 7)),
            layers.Conv2D(64, 8),
            layers.Activation("linear"),
            layers.Conv2D(128, 10),
            layers.Activation("linear"),
            layers.Flatten(),
            layers.Dense(64),
            layers.Activation("sigmoid"),
            layers.Dense(num_actions),
            layers.Activation("linear")
        ]
    )

global model
model = create_q_model()
# model.load_weights('my_weights_v3') # 这个相对路径不知道该咋写
model.load_weights('my_weights_v3.h5')
# model.load_weights('./weight/my_weights_v3.h5')

w = model.get_weights()
# print(w)

with open("weight.txt","w") as file:
    for item in w:
        print(item.shape)
        item = item.flatten()
        item = list(item)
        file.write(str(item))
