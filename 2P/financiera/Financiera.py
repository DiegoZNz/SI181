from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Cargar datos de entrenamiento desde un archivo CSV
datos_entrenamiento = np.genfromtxt("2P\\financiera\datos_clientes.csv", delimiter=",", dtype=np.float32)

# Separar las características y las etiquetas
X_train = datos_entrenamiento[:, 2:]  # Edad e Ingresos
y_train = (datos_entrenamiento[:, 2] >= 25) & (datos_entrenamiento[:, 3] >= 29000)  # Fiable (True o False)
y_train = y_train.astype(int)  # Convertir a 0 y 1

# Normalizar características por separado
scaler_edad = MinMaxScaler()
scaler_ingresos = MinMaxScaler()

X_train[:, 0:1] = scaler_edad.fit_transform(X_train[:, 0:1])
X_train[:, 1:2] = scaler_ingresos.fit_transform(X_train[:, 1:2])

# Crear el modelo
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))  # Capa de entrada
model.add(Dense(8, activation='relu'))  # Capa oculta
model.add(Dense(1, activation='sigmoid'))  # Capa de salida

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)  # verbose=1 para imprimir el progreso


# Cargar datos para predicción desde un archivo CSV
datos_prediccion = np.genfromtxt("2P\\financiera\pred.csv", delimiter=",", dtype=np.float32)

# Preprocesar datos para predicción
X_pred = datos_prediccion[:, 2:]  # Utiliza solo las primeras tres columnas de los datos de predicción

# Normalizar características de predicción
X_pred[:, 0:1] = scaler_edad.transform(X_pred[:, 0:1])
X_pred[:, 1:2] = scaler_ingresos.transform(X_pred[:, 1:2])

# Realizar predicciones utilizando el modelo entrenado
predictions = model.predict(X_pred)

# Redondear las predicciones a 0 o 1
y_pred = np.round(predictions).astype(int)

# Imprimir predicciones
print("Predicciones:")
print(y_pred)
