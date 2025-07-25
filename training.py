#importar librerias
import random
import json
import pickle   #trabajamos con archivos que podemos guardar 
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

#modelo de red neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#cambiar a 1 y 0

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


#lista de palabras, clases, documentos, ignorar_palabras

words = []
classes = []   
documents = []
ignore_letters = ['?', '!', '.', ',', "'s", "'m", "'re", "'ll", "'ve", "'d", "'t"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenizar cada palabra a 1 y 0
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        #en esta lista, relacionar las palabras que añadimos con el indentificador
        documents.append ((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

#palabras sin conjugar
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))  # ordenar palabras y ponerlos en un set

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#creamos lista para hacer entrenamiento
training = []
output_empty = [0]*len(classes)


for document in documents:
    bag= []
    word_patterns = document[0]
    #poner minusculas  
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    #para cada palabra dentro de la lista, añadimos 1 en la lista de word patterns
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    #primer indice de todas las clases, ponemos 1 si la palabra coincide con el tag
    output_row[classes.index(document[1])] = 1
    #añadir a la lista de entrenamiento
    training.append([bag, output_row])

    random.shuffle(training)  #mezclar los datos
#arreglar error

    training = np.array(training, dtype=object)  #convertir a array
    print (training)


#entrenamiento

train_x = list(training[:, 0]) #si pertenece al primer patro
train_y = list(training[:, 1]) #pertenece al segundo patron

#red neuronal secuencial
model = Sequential()
#añadir capas con 128 neuronas y tamaño de entrada
#la longitud de la lista de palabras
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))  #dropout para evitar el sobreajuste
model.add(Dense(64, activation='relu'))  #capa de salida para clasificar
model.add(Dropout(0.5))  #dropout para evitar el sobreajuste
model.add(Dense(len(train_y[0]), activation='softmax'))  #capa de salida para clasificar

sgd= sgd_experimental = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  #compilar el modelo
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)  #entrenar el modelo

#guardar el modelo
model.save('chatbot_model.h5', train_process)  #guardar el modelo
print("Modelo creado")  #mensaje de confirmación