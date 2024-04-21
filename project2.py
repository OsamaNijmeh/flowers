import pandas as pd
data = pd.read_csv("iris.csv")

x = data.drop(columns=["species"])

target_names = data["species"].unique()
target_numbers = {n:i for i,n in enumerate(target_names)}
y = data["species"].map(target_numbers)

from keras.utils.np_utils import to_categorical
yc = to_categorical(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='softmax'))
model.compile(Adam(learning_rate=0.99), loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x,yc,epochs=30)

yp = model.predict(x)

sl = float(input("Sepal Length"))
sw = float(input("Sepal Width"))
pl = float(input("Petal Length"))
pw = float(input("Petal Width"))

import numpy as np
xt = np.array([[sl, sw, pl, pw]])

yt = model.predict(xt)
spc = np.argmax(yt)
spc_numbers = {i:n for i,n in enumerate(target_names)}
print(spc_numbers[spc])