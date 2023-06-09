#OpenCV y TensorFlow
import numpy as np
import csv
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

# Variables necesarias para configurar la carga del modelo a utilizar
longitud, altura = 53, 53
modelo = './model/model.h5'
pesos_modelo = './model/pesos.h5'
cnn = load_model(modelo)        # Carga del modelo entrenado
cnn.load_weights(pesos_modelo)  # Carga de los pesos de la red
    
################ FUNCIONES PARA LA PREDICCION ################
def predict(file):
  global actual_prediction, row
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)   # Convertir la imagen en un arreglo
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)    # Array de dos dimensiones [[0,0...0]] donde la primer dimension contiene la cantidad de clases en el modelo
  # La prediccion arrojara como resultado un 1 en la clase correspondiente
  result = array[0] # La dimension 0 es la que incluye las clases del modelo entrenado
  answer = np.argmax(result)    # Obtiene el indice del elemento que tiene el valor mas alto
  # Proceso de comparacion entre cada indice del modelo entrenado
  print("\nLa CNN dice: ")
  with open("Training_indices.csv") as file:
      reader = csv.reader(file, delimiter=',')
      for row in reader:
          print("{}".format(row[answer]))
 
    ##################### MAIN #####################
while True:
    dir_image='./data/input/A-01.jpg'
    predict(dir_image)
    break