#archivo de predicciones
#importar librerias
import random
import json
import pickle   #trabajamos con archivos que podemos guardar 
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
 
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

#importar archivos que hemos guardado
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))   
model = load_model('chatbot_model.h5') 


#funciones
def clean_up_sentence(sentence):
    #tokenizar la frase
    sentence_words = nltk.word_tokenize(sentence)
    #convertir a minusculas y lematizar
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#convertir palabras a 0 y 1 segun la categoria que pertenezcan
def bag_of_words(sentence):
    #tokenizar la frase
    sentence_words = clean_up_sentence(sentence)
    #inicializar la lista de 0
    bag = [0]*len(words)
    
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    print(bag)
    return np.array(bag) #nos devuelve un array de numpy con estas palabras

def predict_class(sentence):
    bow = bag_of_words(sentence)  #convertir la frase a 0 y 1
    res = model.predict(np.array([bow]))[0]  #devolver una probabilidad de que pertenezca a una categoria
    max_index = np.where(res ==np.max(res))[0][0]  #encontrar el indice de la categoria con mayor probabilidad
    category = classes[max_index]  #obtener la categoria
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = "" 
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        return result
    
def respuesta(message):  
    ints = predict_class(message)  
    res = get_response(ints, intents)  #obtener la respuesta de la categoria
    return res  #imprimir la respuesta

while True:
    message = input()
    print(respuesta(message))