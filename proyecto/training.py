# DEFINICION DE LIBRERIAS
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense#, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
#from tensorflow.keras.optimizers import Adam # Activar solo en caso de implementar TensorFlow < v.2
#import sys

# Cerrar toda sesion activa de Keras para dedicar mayor recurso al entrenamiento
K.clear_session()
tf.compat.v1.disable_eager_execution()

data_entrenamiento = './data/training'
data_validacion = './data/validation'

# PARAMETROS PARA DEFINIR CARACTERISTICAS DE LA RED NEURONAL
epocas=25   # Iteraciones
altura, longitud = 53, 53   # Dimension de la imagen a procesar en pixeles
batch_size = 32 # Cantidad de imagenes recopiladas en cada paso
pasos = 1000    # Cantidad de veces que se procesa la informacion de entrenamiento en cada epoca
validation_steps = 200  # Razon muestral en que se verifica el aprendizaje
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 42
lr = 0.0001 # Learning rate. Ajustes de optimizacion
neuronas = 512
num_of_test_samples = clases*25

# PRE PROCESAMIENTO DE LAS IMAGENES
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,     # Reescalado de pixeles de 0-255 a 0-1
    shear_range=0.0,    # Inclinacion
    zoom_range=0.0,     # Zoom en cada imagen
    horizontal_flip=False)  # Inversion de la imagen

test_datagen = ImageDataGenerator(rescale=1./255)

# PROCESAMIENTO DE LAS IMAGENES DE ENTRENAMIENTO
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

# PROCESAMIENTO DE LAS IMAGENES DE VALIDACION
validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

# CREACION DE LA RED NEURONAL CNN
cnn = Sequential()  # Definicion de una red neuronal multicapa
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu')) # Definicion de la primer capa de convolucion
cnn.add(MaxPooling2D(pool_size=tamano_pool))    # Defincion de la primera capa de MaxPooling

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))   # Definicion de la segunda capa de convolucion
cnn.add(MaxPooling2D(pool_size=tamano_pool))    # Defincion de la segunda capa de MaxPooling

# En este punto la imagen se encuentra a una dimension muy pequeña pero profunda
# La siguiente seccion permitira convertir dicha imagen a una sola dimension con toda
# la informacion de la red
cnn.add(Flatten())
cnn.add(Dense(neuronas, activation='relu'))
# Apagar un porcentaje de neuronas permite adaptar la red al encontrar distintas soluciones
# de reconocimiento. Emplear todas ellas probablemente generara una unica solucion. 
cnn.add(Dropout(0.5))   # Desactiva el porcentaje de las neuronas definidas.
cnn.add(Dense(clases, activation='softmax'))    # Segunda capa densa y ultima capa de la red
# La capa softmax genera el porcentaje de prediccion en cada imagen

cnn.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])  # Que tan bien aprende?
cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

# GENERACION DEL MODELO Y PESOS DE LA CNN
target_dir = './model/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./model/model.h5')    # Estructura de la red neuronal
cnn.save_weights('./model/pesos.h5')    # Pesos generados
print("Modelo CNN creado")

# EXPORTANDO LA LISTA DE CLASES GENERADAS
# Almacenando la organizacion de cada clase
index_training = entrenamiento_generador.class_indices

# Generando un fichero CSV donde se almacene el indice de cada clase creada
Lista = []
Lista.append(index_training)
with open('Training_indices.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, Lista[0].keys())
    writer.writeheader()
    for element in Lista:
        writer.writerow(element)        
print('Fichero CSV creado')

# GENERANDO EL REPORTE DE LA MATRIZ DE CONFUSION
Y_pred = cnn.predict_generator(validacion_generador, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Matriz de confusion')
print(confusion_matrix(validacion_generador.classes, y_pred))
print('Reporte de clasificacion')
target_names = list(entrenamiento_generador.class_indices.keys())
print(classification_report(validacion_generador.classes, y_pred, target_names=target_names))

# GRAFICA DE LA MATRIZ DE CONFUSION
# Utilizar solo si se desea conocer automaticamente la precision de prediccion
fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target_names):
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, metrics.auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return metrics.roc_auc_score(y_test, y_pred, average=average)
validacion_generador.reset()
y_pred = cnn.predict_generator(validacion_generador, verbose = True)
y_pred = np.argmax(y_pred, axis=1)
multiclass_roc_auc_score(validacion_generador.classes, y_pred)

print('\n##############')
print('ENTRENAMIENTO COMPLETADO')
print('##############')