import numpy as np
from keras import Sequential
from keras.layers import Dense

#create random data
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential([
    Dense(1, input_dim=10)
])
model.compile(loss='mse', optimizer='adam')

#train model by 1 iteration
model.fit(test_x, test_y, epochs=1, verbose=False)

#predict
data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
prediction = model.predict(data)
print(prediction)

#save model
from keras2cpp import export_model
export_model(model, 'example.model')