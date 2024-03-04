#crea tu primer mlp en keras
from keras.models import Sequential
from keras.layers import Dense
import numpy 

# fijamos la semilla aleatoria para reproducibilidad
numpy.random.seed(7) #porque son 7 valores

#cargamos los datos
dataset = numpy.loadtxt("2P\diabetes\pima-indians-diabetes.csv", delimiter=",")
# dividimos en variables de entrada (X) y salida (Y)
X = dataset[:,0:8]
Y = dataset[:,8]

#creamos el modelo
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) #capa de entrada
model.add(Dense(8, activation='relu')) #capa oculta
model.add(Dense(1, activation='sigmoid')) #capa de salida

#compilamos el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#ajustamos el modelo
model.fit(X, Y, epochs=150, batch_size=10)

#predicciones
predictions = model.predict(X)
print(predictions)

#redondeamos las predicciones
rounded = [round(x[0]) for x in predictions]
print(rounded)