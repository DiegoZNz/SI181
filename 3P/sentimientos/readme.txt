Para ejecutar este programa, es necesario tener instaladas las siguientes librerías de Python:

pandas: Para el manejo y análisis de datos en formato tabular.
deep-translator: Para la traducción automática de texto.
nltk: Plataforma líder para el procesamiento de lenguaje natural en Python.
textblob: Librería para el procesamiento de datos textuales, incluyendo análisis de sentimientos.
vaderSentiment: Librería específica para el análisis de sentimientos utilizando el algoritmo VADER.
matplotlib: Librería para la visualización de datos en forma de gráficos.

Instalación
Puedes instalar estas librerías utilizando el administrador de paquetes pip. Abre una terminal y ejecuta los siguientes comandos:

pip install pandas
pip install deep-translator
pip install nltk
pip install textblob
pip install vaderSentiment
pip install matplotlib


Después de instalar nltk, es necesario descargar algunos recursos adicionales. Abre una sesión de Python y ejecuta los siguientes comandos:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')


Esto descargará los recursos necesarios para la tokenización, eliminación de stopwords y lematización.
(Ya viene comentado en codigo solo seria descomentarlos, linea 10 y 12)