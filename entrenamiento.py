import cv2
import numpy as np
import os

direccion='C:/Users/mavo0/Desktop/maskDetector/fotos' #direccion donde estan las carpetas con_Cubrebocas y sin_Cubrebocas
listaDir = os.listdir(direccion) #listamos las carpetas que estan dentro de esta direccion
print('lista carpetas: ',listaDir)

etiquetas=[] #etiquetas asociadas a cada imagen conTapabocas=0 y sinTapabocas=1
rostros=[] #arreglo de rostros
etiqueta=0 # 0 y 1

#bucle para iterar en la lista de carpetas ['con_Cubrebocas', 'sin_Cubrebocas']
for nomDir in listaDir:
    rutaDir=direccion + '/' + nomDir
    #bucle para asignar las etiquetas a las fotos
    for nomArchivo in os.listdir(rutaDir):
        rutaImagen=rutaDir+'/'+nomArchivo
        print(rutaImagen)
        imagen = cv2.imread(rutaImagen, 0) #cargamos la imagen de la ruta especificada
        #prueba de que se estan asignando las direcciones correctas
        #cv2.imshow("Image", imagen)
        #cv2.waitKey(10)

        #almacenamos todas las imagenes
        rostros.append(imagen)
        #almacenamos todas las etiquetas
        etiquetas.append(etiqueta)
    etiqueta+=1 #incrementar en uno la etiqueta de tal manera que conTapabocas=0 y sinTapabocas=1

#verificamos si esta funcionando el programa contando el numero de fotos etiquetadas con 1 y 0
print("Etiqueta 0: ", np.count_nonzero(np.array(etiquetas) == 0))
print("Etiqueta 1: ", np.count_nonzero(np.array(etiquetas) == 1))

#creamos un modelo con LBPH FaceRecognizer
deteccion= cv2.face.LBPHFaceRecognizer_create()
#entrenamos nuestro modelo con los rostros que ya tienen asignadas sus etiquetas
deteccion.train(rostros,np.array(etiquetas)) 
print("entrenando")
#almacenamos el modelo obtenido
deteccion.write('maskDetectorModeloLBP.xml')
print("modelo almacenado")
