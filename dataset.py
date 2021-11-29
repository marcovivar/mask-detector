import cv2 #libreria de vision artificial
import os #módulo para crear la carpeta en donde se van a guardar la imágenes de los rostros
from matplotlib import pyplot #va a interactuar con la red neuronal convolucional
import imutils #para operar las imágenes 
from mtcnn_cv2 import MTCNN #es la red neuronal que se va a utilizar para detectar los rostros 

#creamos la carpeta en donde se guardara el entrenamiento
nombre='con_Cubrebocas'
direccion='C:/Users/mavo0/Desktop/maskDetector/fotos'
carpeta=direccion+'/'+nombre

#creamos la carpeta si no existe
if not os.path.exists(carpeta):
  print('Carpeta creada: ',carpeta)
  os.makedirs(carpeta)

#capturamos video en tiempo real

detector=MTCNN() #detector es igual a la red neuronal convolucional 
captura = cv2.VideoCapture(0) #iniciamos camara
conta=0 #como queremos 300 fotos, este contador va a numerar las fotos y nos sacara del bucle cuando se igual a 299
while True:
  ret, imagen = captura.read()   #capturamos video frame a frame
  gris=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
  auxImagen=imagen.copy() #creamos una copia de la imagen
  rostros=detector.detect_faces(auxImagen) #detecta los rostros a traves de la red neuronal convolucional
  #almacenamos las fotos con su respectiva numeracion y con ciertas medidas
  for i in range(len(rostros)):
    x1,y1,ancho,alto=rostros[i]['box'] #verificamos que solo tome los pixeles de nuestro rostros, atraves de un box de cada indice box:red neuronal convolucional,por cada rostro que detecte nos devolvera una caja con los pixeles )
    x2,y2=x1+ancho,y1+alto #definicmo coordenadas para obtener cuadrado
    rostroRegistrado=imagen[y1:y2,x1:x2]
    rostroRegistrado=cv2.resize(rostroRegistrado,(150,200),interpolation=cv2.INTER_CUBIC) #Ajustar el tamaño de las fotos para que siempre queden igual y que pueda haber coinsidencia en el entrenamiento
                                                                                          #se le asigna una dimension, tod esto se va hacer a traves de una interpolacion cubica  
    cv2.imwrite(carpeta+"/rostro{}.jpg".format(conta),rostroRegistrado) #almacenamos las fotos ya redimensionadas
    conta=conta+1
  #mostramos el video tomado en tiempo real
  cv2.imshow('Entrenamiento', imagen)
  #salimos del bucle si presionamos q o si captura las 300 fotos
  q=cv2.waitKey(1)
  if q==27 or conta>=300:
    break
#finalizamos cammara y cerramos ventana
captura.release()
cv2.destroyAllWindows()
