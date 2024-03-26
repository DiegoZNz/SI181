import re
import pandas as pd
from deep_translator import GoogleTranslator
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt

# Cargar el archivo Excel
excel_file = '3P/AnálisisSentimientosGrupal/forms181.xlsx'
mydata = pd.read_excel(excel_file)


# Función para traducir texto de español a inglés
def translate_text(text):
    return GoogleTranslator(source="es", target="en").translate(text)

# Aplicar la traducción a la columnas
mydata['pr1'] = mydata['pr1'].apply(translate_text)
mydata['pr2'] = mydata['pr2'].apply(translate_text)
print(mydata)

# Función para limpiar el texto del archivo
def clean(text):
    # Removemos los caracteres y números que no se ocuparon
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

# Limpiar el texto en las columnas traducidas
mydata['pr1'] = mydata['pr1'].apply(clean)
mydata['pr2'] = mydata['pr2'].apply(clean)
print(mydata)

# Función para tokenizar y etiquetar el texto
pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

# Función para lematizar el texto
wordnet_lemmatizer = WordNetLemmatizer()
def lematize(pos_data):
    lemma_rew = ""
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

# Tokenizar y etiquetar el texto en las columnas limpiadas
mydata['POS_tagged_pr1'] = mydata['pr1'].apply(token_stop_pos)
mydata['POS_tagged_pr2'] = mydata['pr2'].apply(token_stop_pos)

# Lematizar el texto tokenizado y etiquetado
mydata['Lemma_pr1'] = mydata['POS_tagged_pr1'].apply(lematize)
mydata['Lemma_pr2'] = mydata['POS_tagged_pr2'].apply(lematize)

# Mostrar el DataFrame resultante
print(mydata[['pr1', 'Lemma_pr1']])
print(mydata[['pr2', 'Lemma_pr2']])

# Función para calcular subjetividad
def getSubjectivity(comentarios):
    return TextBlob(comentarios).sentiment.subjectivity

# Función para caluclar polaridad
def getPolarity(comentarios):
    return TextBlob(comentarios).sentiment.polarity

# Función para analizar los resultados
def analysis(score):
    if score < 0:
        return 'Negativo'
    elif score == 0:
        return 'Neutro'
    else:
        return 'Positivo'

# Crear un DataFrame para el análisis final
fin_data_pr1 = pd.DataFrame(mydata[['pr1', 'Lemma_pr1']])
fin_data_pr2 = pd.DataFrame(mydata[['pr2', 'Lemma_pr2']])

# Calcular subjetividad y polaridad para pr1 y pr2
fin_data_pr1['Subjetividad'] = fin_data_pr1['Lemma_pr1'].apply(getSubjectivity)
fin_data_pr1['Polaridad'] = fin_data_pr1['Lemma_pr1'].apply(getPolarity)
fin_data_pr1['Resultado'] = fin_data_pr1['Polaridad'].apply(analysis)

fin_data_pr2['Subjetividad'] = fin_data_pr2['Lemma_pr2'].apply(getSubjectivity)
fin_data_pr2['Polaridad'] = fin_data_pr2['Lemma_pr2'].apply(getPolarity)
fin_data_pr2['Resultado'] = fin_data_pr2['Polaridad'].apply(analysis)

# Imprimir el DataFrame resultante para pr1 y pr2
print("Resultado para Describe tu experiencia en las clases de los martes y jueves este cuatrimestre:")
print(fin_data_pr1)
print("\nResultado para ¿Qué emociones experimentas al pensar en la clase del profesor Nelson?:")
print(fin_data_pr2)

# Gráfica de los resultados para pr1
print("Gráfica de los resultados para Describe tu experiencia en las clases de los martes y jueves este cuatrimestre:")
tb_counts_pr1 = fin_data_pr1['Resultado'].value_counts()
print(tb_counts_pr1)

plt.figure(figsize=(10,7))
plt.title("TextBlob Describe tu experiencia en las clases de los martes y jueves este cuatrimestre")
plt.pie(tb_counts_pr1.values, labels=tb_counts_pr1.index, explode=(0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.show()

# Gráfica de los resultados para pr2
print("Gráfica de los resultados para ¿Qué emociones experimentas al pensar en la clase del profesor Nelson?:")
tb_counts_pr2 = fin_data_pr2['Resultado'].value_counts()
print(tb_counts_pr2)

plt.figure(figsize=(10,7))
plt.title("TextBlob ¿Qué emociones experimentas al pensar en la clase del profesor Nelson?")
plt.pie(tb_counts_pr2.values, labels=tb_counts_pr2.index, explode=(0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.show()
