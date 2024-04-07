import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np
import random 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#importamos y cargamos el archivo JSON
randomwords = []
words = []
classes = []
documents = []
ignore_words = ['¿', '?', '!']

data_file = open('3P\chatbot\intents.json', encoding='utf-8').read()
intents = json.loads(data_file)
print(intents)

#preprocesamos los datos
#creamos los tokens
#iteramos a través de los patrones y tokenizamos las palabras
#y agregamos cada palabra a la lista de palabras
#nuestras etiquetas

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenizamos cada palabra
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #agregamos los documentos en el corpus
        documents.append((w, intent['tag']))
        #agregamos a nuestra lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
#lemmatizamos, bajamos a minúsculas y eliminamos duplicados
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
print(words)

pickle.dump(classes,open('3P\chatbot\classes.pkl','wb'))
pickle.dump(words,open('3P\chatbot\words.pkl','wb'))

#creamos nuestro dataset de entrenamiento
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    print(training)

random.shuffle(training)
train_x = [t[0] for t in training]
train_y = [t[1] for t in training]

#Se crea el modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('3P\chatbot\chatbot_model.h5', hist)