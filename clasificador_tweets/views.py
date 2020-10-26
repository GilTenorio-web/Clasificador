from django.shortcuts import render
from django.template import Template, Context #Necesario para usar Template y Context
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from django.http import HttpResponse
# Create your views here.

class Clasificador(object):
    #Abrimos el archivo de entrenamiento. 
    docExterno = open("Corona_NLP_train.csv",encoding="ISO-8859-1")
    covid = pd.read_csv(docExterno, encoding="ISO-8859-1")
    #docExterno.close()
    #Obtenemos las etiquetas de clasificación
    colEtiquetas = covid.Sentiment
    #Obtenemos los tweets del archivo de prueba. 
    colTweet = covid.OriginalTweet
    #Mostramos los tipos de etiquetas
    tiposEtiquetas = set(colEtiquetas)
    #Este objeto es global porque lo ocupa la vista para poder obtener os atributos y mostrarlos en pantalla
    vec = TfidfVectorizer(stop_words="english", min_df = 2)
    #Se tienen que particionar los datos para que el algoritmo pueda aprender  
    #Particionamos los datos del archivo de entrenamineto para obtener datos de entrenamiento y de prueba.
    #Para particionarlo utilizamos train_test_split(datos, etiquetas y estado ramdon)
    def particionar(self):
        #Este metodo particiona los datos y devuelve una lista con las variables resultantes
        xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(self.colTweet, self.colEtiquetas, random_state=42)
        datos = [xEntrenamiento,xPrueba,yEntrenamiento,yPrueba]
        return datos

    #Una vez que tengamos los datos se procede hacer una matriz de entrenamiento y una matriz de prueba
    #con el empleo de TfidVectorizer, fit_transform para los datos de entrenamiento y utilizamos transform para los datos prueba.
    #Para esto hay dos versiones: Con el objeto TfidfVectorizer como atributo global o como atributo propio del metodo 
    def creaMatriz(self, dato):
        #Este metodo crea matrices. Una con datos de entrenamiento y otra con datos de prueba.
        #vec = TfidfVectorizer(stop_words="english", min_df = 2)
        matriz = self.vec.fit_transform(dato) 
        #Si yo pongo aqui los atributos no tengo manera de imprimirlo. Solo se mostrarian en consola. 
        return matriz
      
    def creaMatrizPrueba(self, dato):
        #Este metodo crea  una matriz datos de prueba.
        matriz = self.vec.transform(dato) 
        
        return matriz

    def entrenarModelo(self,matrizE, datos):
        #Se crea una instancia 
        nb = MultinomialNB()
        #Se entrena el algoritmo con (matrix de entrenamiento, target)
        nb.fit(matrizE, datos[2])
        return nb

    def predecir(self,nb,matriz):
        y_pred = nb.predict(matriz)
        return  y_pred
    

def clasificador(request):

  
    
    c2 = Clasificador() #Creamos una instancia de Clasificador()
    datos = c2.particionar()#Se particionan los datos
    
    matrizEntrenamiento = c2.creaMatriz(datos[0]) #Esta matriz es para el model0
    atributosE = c2.vec.get_feature_names() #Obtenemos los atributos de la matriz
    #pd.DataFrame(matrizEntrenamiento.toarray(), columns=atributosE).head(10)
    matrizPrueba = c2.creaMatrizPrueba(datos[1]) #Esta matriz es para el modelo
    atributosP = c2.vec.get_feature_names() #Obtenemos los atributos de la matriz
    #pd.DataFrame(matrizPrueba.toarray(), columns= atributosP).head(10)
    nb = c2.entrenarModelo(matrizEntrenamiento, datos)
    y_pred = c2.predecir(nb,matrizPrueba)
    
    #Obtenemos del request lo enviado por el formulario
    tweets = request.GET["tweet"]
    #Esto lo ingresamos en una matriz
    tweets_nuevos = [tweets] 
    matrizNuevosTweets = c2.creaMatrizPrueba(tweets_nuevos)

    #Realizamos la nueva prediccion
    nuevaPrediccion = c2.predecir(nb,matrizNuevosTweets)
    
    contexto = {"cov":c2.covid, "tweets":c2.colTweet, "etiquetas":c2.colEtiquetas, "tiposEtiquetas":c2.tiposEtiquetas,
                        "xEntrenamiento":datos[0],"xPrueba":datos[1],"yEntrenamiento":datos[2],"yPrueba":datos[3],
                        "matrizE":matrizEntrenamiento, "matrizP":matrizPrueba, "data1":atributosE, "data2":atributosP,
                        'y_pred': y_pred, 'nuevosTweets':tweets_nuevos,'nuevaPrediccion':nuevaPrediccion}
 
    return render(request,"clasificador_tweets.html", contexto)

def home(request):


    return render(request,"home.html")



