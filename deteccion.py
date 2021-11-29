import cv2 #libreria de vision artificial
import os #módulo para crear la carpeta en donde se van a guardar la imágenes de los rostros
from mtcnn_cv2 import MTCNN #es la red neuronal que se va a utilizar para detectar los rostros 


direccion='C:/Users/mavo0/Desktop/maskDetector/fotos' #direccion donde estan los dataset
listaDir = os.listdir(direccion) #listamos las carpetas que estan dentro de esta direccion
print("Lista de carpetas: ",listaDir)


#llamamos al modelo de deteccion
deteccion=cv2.face.LBPHFaceRecognizer_create()
#leemos el modelo
deteccion.read("maskDetectorModeloLBP.xml")

#capturamos video en tiempo real

detector=MTCNN() #detector es igual a la red neuronal convolucional(creamos el objeto que va a detectar) 
captura = cv2.VideoCapture(0) #iniciamos camara

while True:
  ret, imagen = captura.read()   #capturamos video frame a frame
  if ret==False:break
  gris = cv2.cvtColor (imagen, cv2.COLOR_BGR2GRAY) #Convertir a escala de grises 
  auxImagen=imagen.copy() #creamos una copia de la imagen
  auxImagen2=gris.copy()
  rostros=detector.detect_faces(auxImagen) #detecta los rostros a traves de la red neuronal convolucional
  #iteramos en rostros
  for i in range(len(rostros)):
    x1,y1,ancho,alto=rostros[i]['box'] #verificamos que solo tome los pixeles de nuestro rostros, atraves de un box de cada indice box:red neuronal convolucional,por cada rostro que detecte nos devolvera una caja con los pixeles )
    x2,y2=x1+ancho,y1+alto #definicmo coordenadas para obtener cuadrado
    rostroRegistrado=auxImagen2[y1:y2,x1:x2]
    rostroRegistrado=cv2.resize(rostroRegistrado,(150,200),interpolation=cv2.INTER_CUBIC) #Ajustar el tamaño de las fotos para que siempre queden igual y que pueda haber coinsidencia en el entrenamiento
                                                                                          #se le asigna una dimension, tod esto se va hacer a traves de una interpolacion cubica  
    resultado=deteccion.predict(rostroRegistrado) #este metodo nos hara una prediccion si trae o no trae cubrebocas y se almacenara en resultado (1 o 0) 
    
    #mostramos en pantalla los resultados
    if resultado[0]==0:
      cv2.putText(imagen, "{}".format(listaDir[0]), (x1, y1 - 15), 2, 1, (0,255,0), 1, cv2.LINE_AA)
      cv2.rectangle(imagen, (x1, y1), (x1+ancho, y1 + alto), (0,255,0), 2)
    if resultado[0]==1:
      cv2.putText(imagen, "{}".format(listaDir[1]), (x1, y1 - 15), 2, 1, (0,0,255), 1, cv2.LINE_AA)
      cv2.rectangle(imagen, (x1, y1), (x1+ancho, y1 + alto), (0,0,255), 2)  
  #mostramos el video tomado en tiempo real
  cv2.imshow('maskDetector', imagen)
  #salimos del bucle si presionamos q
  q=cv2.waitKey(1)
  if q==27 :
    break
#finalizamos cammara y cerramos ventana
captura.release()
cv2.destroyAllWindows()
