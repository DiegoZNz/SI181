import re
import pandas as pd
from deep_translator import GoogleTranslator
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk import pos_tag
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('wardnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.corpus import sentiwordnet as swn
import matplotlib.pyplot as plt


#Paso1: Limpieza y traducción de texto

#cargar el archivo
mydata = pd.read_csv('3P/sentimientos/comentarios.csv',delimiter=',')
mydata.head()#head() muestra las primeras 5 filas
print(mydata)
#Traducir el texto a inglés
translator = GoogleTranslator(source="es", target="en")
mydata.review=mydata.review.apply(translator.translate)
print(mydata)

#Funcion para limpiar el texto del archivo
def clean(text):
    #Removemos los caracteres y numeros que no se ocuparon
    text = re.sub('[^A-Za-z]+',' ',text)
    return text

#Limpiamos el texto en la columna comentario
mydata['Cleaned_Reviews'] = mydata['review'].apply(clean)
mydata.head()
print(mydata)

#Paso 2:
#2.1 Tokenización = Es el proceso para dividir el texto en diferentes partes llamadas tokens
#2.2 Etiquetado POS (Etiquetado gramatical): Es un proceso de conversion de cada token en una tupla que tiene la forma (palabra, etiqueta)
#2.3 Eliminación de palabras irrelevantes

#Generamos el diccionario de etiquetado gramatical / Tokenizacion
# J = Adjetivo V = Verbo N = Sustantivo y R = Adverbio
pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

#Definimos la funcion para tokenizar y etiquetar el texto
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

mydata['POS_tagged'] = mydata['Cleaned_Reviews'].apply(token_stop_pos)
mydata.head()
print(mydata)

#Paso 3: Obtencion de la palabra raíz
#Una palabra es parte de una palabra responsable de su significado lexico
wordnet_lemmatizer = WordNetLemmatizer()

def lematize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

mydata['Lemma'] = mydata['POS_tagged'].apply(lematize)
mydata.head()

#Aquí se muestra la oración inicial, contra la oracion procesada
#Mostrando las palabras importantes que se procesaran
print(mydata[['review', 'Lemma']])


#Paso 4: Utilizar los algoritmos de análisis de sentimientos (textblob, vader, etc)

#Primer libreria textblob

#Esto nos calculara la polaridad, esta varia de -1 a 1 (1 mas positivo, 0 es neutral, -1 mas negativo)
#La subjetividad (Efectividad), esta varia de 0 a 1 (0 es objetivo y 1 muy subjetivo)

#Funci[on para calcular subjetividad
def getSubjectivity(comentarios):
    return TextBlob(comentarios).sentiment.subjectivity

#Funcion para caluclar polaridad
def getPolarity(comentarios):
    return TextBlob(comentarios).sentiment.polarity

#Funcion para analizar los resultados
def analysis (score):
    if score < 0:
        return 'Negativo'
    elif score == 0:
        return 'Neutro'
    else:
        return 'Positivo'
    
fin_data = pd.DataFrame(mydata[['review', 'Lemma']])
fin_data['Subjetividad'] = fin_data['Lemma'].apply(getSubjectivity)
fin_data['Polaridad'] = fin_data['Lemma'].apply(getPolarity)
fin_data['Resultado'] = fin_data['Polaridad'].apply(analysis)
fin_data.head()
print(fin_data)

tb_counts=fin_data['Resultado'].value_counts()
print(tb_counts)

plt.figure(figsize=(10,7))
plt.title("Resultado TextBlob")
plt.pie(tb_counts.values,labels=tb_counts.index,explode=(0.1,0,0),autopct='%1.1f%%',shadow=False)
plt.show()
#Segunda libreria VADER

#Algoritmo de Sentimiento con Vader (Valance Aware Dictionary and Sentiment Reasoner)

#Este aloritmo aparte de conocer si es positivo, negativo o neutro, tambien nos obtiene la intensidad de la emocion

#La suma de las intensidades de positivo,egativo y neutro nos dara 1.
#El compuesto varia de -1 a 1, y es la metrica utilizada para dibujar el senitmiento

#Regla es positivo si compuesto >= 0.05, neutro si esta en -0.5 < compuesto < 0.5 negativo si -0.5 >= compuesto

analyzer = SentimentIntensityAnalyzer()

#Funcion para calcular los sentimientos con Vader

def vadersentimentanalysis(comentario):
    vs = analyzer.polarity_scores(comentario)
    return vs['compound']

fin_data['Vader_Sentiment'] = fin_data['Lemma'].apply(vadersentimentanalysis)

def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positivo'
    elif compound <= -0.5:
        return 'Neutro'
    else:
        return 'Negativo'
    
fin_data['Vader_Analysis'] = fin_data['Vader_Sentiment'].apply(vader_analysis)
fin_data.head()
print(fin_data)

vader_counts=fin_data['Vader_Analysis'].value_counts()
print(vader_counts)

plt.figure(figsize=(10,7))
plt.title("Resultado Vader_Analysis")
plt.pie(vader_counts.values,labels=vader_counts.index,explode=(0.1,0,0),autopct='%1.1f%%',shadow=False)
plt.show()

#Tercer libreria: SentiWordNet 
#Para esta libreria es importante obtner el POS, Lemma de cada palabra
#Si la puntuacion positiva > puntuacion negativa, el sentimiento es positivo
#Si la puntuacion negativa > puntuacion positiva, el sentimiento es negativo
#Si la puntuacion positiva = puntuacion negativa, el sentimiento es neutral

#Generamos la funcion para analizar los sentimientos

def sentiwordnetanalysis(pos_data):
    sentiment = 0
    tokens_count = 0
    for word, pos in pos_data:
        if not pos:
            continue
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        if not lemma:
            continue
        synsets = wordnet.synsets(lemma, pos=pos)
        if not synsets:
            continue
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1
        print(swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score())
    if not tokens_count:
        return 0
    if sentiment > 0:
        return 'Positivo'
    if sentiment == 0:
        return 'Neutral'
    else:
        return 'Negativo'
    

fin_data['SWN_Analysis'] = mydata['POS_tagged'].apply(sentiwordnetanalysis)
print(fin_data['SWN_Analysis'])
fin_data.head()
print(fin_data)


swn_counts=fin_data['SWN_Analysis'].value_counts()
print(swn_counts)

explode = [0] * len(swn_counts.index)  # Crea una lista de ceros de la longitud adecuada
explode[swn_counts.index.get_loc('Positivo')] = 0.1  # Establece el valor de "explotar" para "Positivo"

plt.figure(figsize=(10,7))
plt.title("Resultados SentiWordNet")
plt.pie(swn_counts.values, labels=swn_counts.index, explode=explode, autopct='%1.1f%%', shadow=False)
plt.show()