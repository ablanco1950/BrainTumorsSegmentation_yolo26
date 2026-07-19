dirname="Brain tumor segmentation.v1i.yolov8-obb\\test\\images"
dirnameLabels="Brain tumor segmentation.v1i.yolov8-obb\\test\\labels"

# 1. Definir diccionario de colores (BGR) y nombres por ID de clase
CLASES = {
    0: {"name": "Tumor", "color": (0, 255, 0)},    # Green
    1: {"name": "No tumor", "color": (255, 0, 0)}       # Blue
    
}

from ultralytics import YOLO

# 1. Cargar modelo e inferir (sin guardar automáticamente)
#model = YOLO("runs/segment/entrenamientos_yolo26/segmentacion_custom-3/weights/best.pt")
model = YOLO("best.pt")

import cv2
import numpy as np

import matplotlib.pyplot as plt
import os
import re



########################################################################
def loadimages(dirname):
 
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
     TabFilePath=[]
   
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 TabFilePath.append(filepath)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return TabFilePath, TabFileName
########################################################################
def loadlabels(dirnameLabels):
 
     imgpath = dirnameLabels + "\\"
     
     Labels = []
     TabFileLabelsName=[]
     Tabxyxy=[]
     TabClass_id=[]
     ContLabels=0
     ContNoLabels=0
         
     print("Reading labels from ",imgpath)
        
     for root, dirnames, filenames in os.walk(imgpath):
         
         for filename in filenames:
                           
                 filepath = os.path.join(root, filename)
                
                 f=open(filepath,"r")

                 #Label=""
                 xyxy=""
                 TabLinxyxy=[]
                 TabLinClass_id=[]
                 for linea in f:
                     
                      Class_id=int(linea[0])
                      TabLinClass_id.append(Class_id)
                      
                      xyxy=linea[2:]
                      xyxy=np.array(xyxy)
                      TabLinxyxy.append(xyxy)
                      
                      
                                            
                 
                 Tabxyxy.append(TabLinxyxy)
                 TabClass_id.append(TabLinClass_id)
     
     return  Tabxyxy, TabClass_id

TabFilePath, TabFileName=loadimages(dirname)
Tabxyxy, TabClass_id =loadlabels(dirnameLabels)

#print(TabFileName)


for i in range(len(TabFileName)):

    #print(TabFilePath[i])
    #print(Tabxyxy[i])
    #print(TabClass_id[i])
    
    
    
    # Crear una capa transparente para el relleno del polígono
    imagen_ori=cv2.imread(TabFilePath[i])
    capa_relleno_labeled = imagen_ori.copy()
    for poligonoLabeled in Tabxyxy[i]:

                print("**************")
                print(TabFileName[i])
                print("**************")
               
                # Dimensiones de tu imagen (ajusta a tus valores reales)
                alto_imagen, ancho_imagen = 640, 640  

                # 1. Extraer el texto puro de tu matriz NumPy de un solo elemento
                # Usar .item() extrae la cadena de texto oculta dentro del objeto np.str_
                texto_puro = poligonoLabeled.item()

                # 2. Separar el texto por sus espacios y convertir cada elemento a decimal (float)
                coordenadas_list = [float(x) for x in texto_puro.strip().split()]

                # 3. Agrupar en parejas de coordenadas [X, Y]
                puntos_normalizados = np.array(coordenadas_list).reshape(-1, 2)

                # 4. Desnormalizar multiplicando por el tamaño de la imagen
                puntos_en_pixeles = puntos_normalizados * [ancho_imagen, alto_imagen]

                # 5. Convertir a enteros de 32 bits para OpenCV
                puntos = puntos_en_pixeles.astype(np.int32)

                # 6. Cambiar la forma a 3D para cv2.polylines o cv2.fillPoly
                puntos = puntos.reshape((-1, 1, 2))

                if TabClass_id[i][0]==0:  
                         color_bgr = (0, 255, 0) # tumor green (puedes cambiarlo a tu gusto)
                         print ("LABELED TUMOR IN IMAGE " + TabFileName[i])
                else:
                    
                    color_bgr = (255, 0, 0) # no tumor blue (puedes cambiarlo a tu gusto)
                    print ("LABELED NO TUMOR IN IMAGE " + TabFileName[i])
                    
                grosor_linea = 2

                imagenOri=cv2.imread(TabFilePath[i])
                # Dibujar el contorno del polígono
                
                # some times appears a rectangle because was labeled with only four points  
                cv2.polylines(imagenOri, [puntos], isClosed=True, color=color_bgr, thickness=grosor_linea)
                
                #cv2.imshow("Labeled",imagenOri)
                #cv2.waitKey(0)
                
                # Rellenar el polígono en la capa transparente
                cv2.fillPoly(imagenOri, [puntos], color=color_bgr)
                

    # 4. Fusionar la imagen original con la capa de relleno (Efecto de transparencia)
    # 0.4 es la opacidad del relleno (40%) y 0.6 la de la imagen original
    opacidad = 0.4
    cv2.addWeighted(capa_relleno_labeled, opacidad, imagenOri, 1 - opacidad, 0, imagenOri)
    #cv2.imshow("Labeled",imagenOri)
    #cv2.waitKey(0)
    
    
    results = model.predict(source=TabFilePath[i], save=False)

    # 2. Cargar la imagen original con OpenCV para dibujar sobre ella
    # Se usa results[0].orig_img para asegurar que coincide exactamente con la matriz original
    imagen_dibujo = results[0].orig_img.copy()

    # Crear una capa transparente para el relleno del polígono
    capa_relleno = imagen_dibujo.copy()

    # 3. Iterar sobre los resultados
    for r in results:
        if r.masks is not None and r.boxes is not None:
            # Extraer las coordenadas en píxeles
            confidence = r.boxes.conf.cpu().numpy()
            clases_id = r.boxes.cls.cpu().numpy().astype(int)
            #confidence=confidence[0]
            if clases_id[0]==0:  
                         color_bgr = (0, 255, 0) # tumor green (puedes cambiarlo a tu gusto)
                         print ("PREDICTED TUMOR IN IMAGE " + TabFileName[i])
            else:
                    
                    color_bgr = (255, 0, 0) # no tumor blue (puedes cambiarlo a tu gusto)
                    print ("PREDICTED NO TUMOR IN IMAGE " + TabFileName[i])
                    
                           
            
            poligonos_pixel = r.masks.xy

            for j in range(len(clases_id)):
                  class_id = clases_id[j]
                  confianza = confidence[j]  # <-- AQUÍ ESTÁ TU VALOR DE CONFIANZA
                  puntos = poligonos_pixel[j].astype(np.int32) # Polígono listo para cv2.polylines

                  puntos = puntos.reshape((-1, 1, 2)) # -1 adapta auomaticamente al numero de columnas
                  
                  info_clase = CLASES.get(class_id, {"name": "Desconocido", "color": (255, 255, 255)})
                  nombre = info_clase["name"]
                  color = info_clase["color"]
                  # 3. Dibujar el polígono con el color de la clase
                  cv2.polylines(imagen_dibujo, [puntos], isClosed=True, color=color, thickness=3)

                  # 4. Preparar el texto (Etiqueta + Porcentaje)
                  texto = f"{nombre}: {confianza * 100:.1f}%"

                  # 5. Encontrar el punto más alto del polígono para poner el texto encima
                  # Buscamos la coordenada Y mínima (la más alta en la pantalla)
                  x_min = int(np.min(puntos[:, :, 0]))
                  y_min = int(np.min(puntos[:, :, 1]))

                  # Ajustar la posición para que el texto no quede pegado al borde del polígono
                  posicion_texto = (x_min, y_min - 10)
                  
                  # Ajustar la posición para que el texto no quede pegado al borde del polígono
                  posicion_texto = (x_min, y_min - 10)

                  # 6. Dibujar un fondo para el texto (mejora la legibilidad)
                  (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                  cv2.rectangle(imagen_dibujo, (x_min, y_min - h - 15), (x_min + w, y_min - 5), color, -1)

                  # 7. Escribir el texto sobre el fondo
                  #cv2.putText(imagen_dibujo, texto, posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


                  # Configuración del texto
                  
                  fuente = cv2.FONT_HERSHEY_SIMPLEX
                  escala = 1
                  grosor_borde = 5
                  grosor_texto = 2

                  # 1. Dibujar el contorno/borde (Negro)
                  cv2.putText(imagen_dibujo, texto, posicion_texto, fuente, escala, (0, 0, 0), grosor_borde, cv2.LINE_AA)

                  # 2. Dibujar el texto principal (Blanco) encima
                  cv2.putText(imagen_dibujo, texto, posicion_texto, fuente, escala, (255, 255, 255), grosor_texto, cv2.LINE_AA)
            
            
    # 4. Fusionar la imagen original con la capa de relleno (Efecto de transparencia)
    # 0.4 es la opacidad del relleno (40%) y 0.6 la de la imagen original
    opacidad = 0.4
    cv2.addWeighted(capa_relleno, opacidad, imagen_dibujo, 1 - opacidad, 0, imagen_dibujo)
    #cv2.imshow("imagen",imagen_dibujo)
    #cv2.waitKey(0)

    fig, axs = plt.subplots(1,2, figsize=(15,5))
    axs[0].imshow(imagenOri);      axs[0].set_title('Labeled: ' + TabFileName[i]);    axs[0].axis('off')
    axs[1].imshow(imagen_dibujo); axs[1].set_title('  Predicted: ') ; axs[1].axis('off')
    #plt.tight_layout();
    plt.show()

    # 5. Guardar o mostrar el resultado final
    #cv2.imwrite("resultado_personalizado.jpg", imagen_dibujo)
    #print("¡Imagen guardada con éxito como 'resultado_personalizado.jpg'!")

