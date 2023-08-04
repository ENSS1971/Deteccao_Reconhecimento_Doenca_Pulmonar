#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importação dos pacotes das bibliiotecas
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#Rede Neural Convulucionais
#Trabalhar com Deep Learning e Redes Neurais 
import tensorflow as tf
# layers - camadas das redes neurais / optimizers - algoritmos ajuste de pesos
from tensorflow.keras import layers, optimizers
#application (pacotes Modelo da rede neural
from tensorflow.keras.applications import ResNet50
#Camadas Input/Entrada Dense/Interligação entre os neorônios 
#AvarangePoolin2D/Dropout/Flatten serão vistos posteriormente
from tensorflow.keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten
#Criando a rede neural
from tensorflow.keras.models import Model
#Salvar os pesos da rede neural
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Salvar os pesos da rede neural
from tensorflow.keras.callbacks import ModelCheckpoint


# In[2]:


#Carregando o diretório das imagens para treinamento
train_data = Path('/Users/eduardo.sepulveda/IA_EXPERT/Medico/Dataset')
#print(train_data.absolute())
os.listdir(train_data)


# In[3]:


#Carregando o diretório das imagens para teste
test_data = Path('/Users/eduardo.sepulveda/IA_EXPERT/Medico/Test')
##print(test_data.absolute())
os.listdir(test_data)


# In[4]:


#Préprocessamento da imagem alterando a escala da imagem
#alterando os valores dos pixels para ficar entre 0 e 1 para que a rede neural possa processar mais rápido as imagens
image_generator = ImageDataGenerator(
    rescale=1./255,
    #rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #fill_mode = 'nearest',
    #horizontal_flip=True,
    validation_split=0.20)


# In[5]:


#Criando subconjunto de treinamento
img_height, img_width = 256, 256

train_generator = image_generator.flow_from_directory(directory=train_data,
                                                      target_size = (img_height, img_width),
                                                      color_mode = 'rgb',
                                                      batch_size = 40,
                                                      class_mode = 'categorical',
                                                      subset = 'training',
                                                      shuffle = True,
                                                      seed = 42,
                                                      interpolation='nearest')


# In[6]:


#Crianddo subconjunto de validação
valid_generator = image_generator.flow_from_directory(directory =train_data,
                                                      target_size = (img_height, img_width),
                                                      color_mode = 'rgb',
                                                      batch_size = 40,
                                                      class_mode = 'categorical',
                                                      subset = 'validation',
                                                      shuffle = True,
                                                      seed = 42,
                                                      interpolation='nearest')


# In[7]:


#Dimensão das imagens
train_images, train_labels = next(train_generator)
train_images.shape


# In[8]:


y_train_imagens, y_train_lables = next(valid_generator)
y_train_imagens.shape


# In[9]:


#Rótulo de cada imagem na base de dados de treinamento
train_labels


# In[10]:


#Nomeando as classes
#Covid-19  - 1 0 0 0
#Normal    - 0 1 0 0
#Viral     - 0 0 1 0
#Bacterial - 0 0 0 1
#Para saber qual classe a imagem pertence analisa-se o valor de saída de cada neurônio
labels_names = {0:'Covid-19', 1:'Normal', 2:'Pneumonia Viral', 3:'Pneumonia Bacterial'}


# In[11]:


#Visualizando as imagens
#Criando uma figura com 6 linhas e 6 colunas
fig, axes = plt.subplots(6, 6, figsize=(12,12))
#Transformando uma matriz em vetores
axes = axes.ravel()
#Criando uma array com 36 indices
for i in np.arange(0, 36):
    #Visualizando as imagens no subgráfico
    axes[i].imshow(train_images[i])
    #Inserindo as lables em cada imagem
    axes[i].set_title(labels_names[np.argmax(train_labels[i])])
    #Removendo as informações dos eixos no gráfico
    axes[i].axis('off')
    #Inserindo espaço na horizontal entre os gráficos
    axes[i].set_title(labels_names[np.argmax(train_labels[i])])


# In[12]:


#Carregamento da rede neural com pesos pré-terinados
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor = Input(shape=(256,256,3)))


# In[13]:


#Visualizar as camadas "Convolution Layers"
#Arquitetura da ResNet já pré-definida
base_model.summary()


# In[14]:


#Quantidade de camadas
len(base_model.layers)


# In[15]:


#Congelar todos os pesos recebidos do download
#Excluir as ultimas 10 layer para podermos treiná-las com as nossas imagens
for layer in base_model.layers[:-10]:
    layers.trainable = False


# In[16]:


#Estrutura da rede neural
head_model = base_model.output
#Aplica a redução da dimensionalidade. Vai realizar a média referente a ùltima camada
head_model = AveragePooling2D()(head_model)
#Converte a matriz em vetor
head_model = Flatten()(head_model)
#Definindo o número de neurônios (128/256)
head_model = Dense(256, activation = 'relu')(head_model)
#Evirar o overfitting - zerando 20% dos neurônios
head_model = Dropout(0.3)(head_model)

head_model = Dense(256, activation = 'relu')(head_model)
#Evirar o overfitting - zerando 20% dos neurônios
head_model = Dropout(0.3)(head_model)
#Definição do números de neurônios na camada de saida da rede neural
#Quatro neurônios porque temos quatro classes (função=softmax)
head_model = Dense(4, activation='softmax')(head_model)


# In[17]:


#Criando o modelo
model = Model(inputs = base_model.input, outputs=head_model)


# In[18]:


model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.RMSprop(learning_rate=0.001, weight_decay=0.005, rho=0.9),
             metrics=['accuracy'])


# In[19]:


#Quantidade de camadas
len(model.layers)


# In[20]:


#Visualizar as camadas "Convolution Layers"
#Arquitetura da ResNet redefinida
model.summary()


# In[21]:


#Salvando o melhor modelo (as melhores métricas)
checkpoint = ModelCheckpoint(filepath = 'weights_md4.hdf5', save_best_only = True)


# In[22]:


#Criando subconjunto de treinamento
img_height, img_width = 256, 256
batch_size = 4

train_generator = image_generator.flow_from_directory(directory=train_data,
                                                      target_size = (img_height, img_width),
                                                      color_mode = 'rgb',
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical',
                                                      subset = 'training',
                                                      shuffle = True,
                                                      seed = 42,
                                                      interpolation='nearest')


# In[23]:


#Crianddo subconjunto de validação
valid_generator = image_generator.flow_from_directory(directory =train_data,
                                                      target_size = (img_height, img_width),
                                                      color_mode = 'rgb',
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical',
                                                      subset = 'validation',
                                                      shuffle = True,
                                                      seed = 42,
                                                      interpolation='nearest')


# In[24]:


#Histórico do treinamento
history = model.fit(train_generator,
                    validation_data = valid_generator,
                    steps_per_epoch = None,
                    epochs=25,
                    #verbose = 0,
                    callbacks = [checkpoint])


# In[25]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

plt.title("Loss")
plt.legend(loc='best')
plt.show()


# In[26]:


#Avaliação da rede natural
history.history.keys()


# In[27]:


#Gerando o gráfico history
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Erro e Taxa de Acerto Durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Taxa de Acerto e Erro')
plt.legend(['Taxa de Acerto','Erro']);


# In[28]:


#Carregando o diretório das imagens para teste
test_directory = Path('/Users/eduardo.sepulveda/IA_EXPERT/Medico/Test')

os.listdir(test_directory)


# In[51]:


#Gerando o teste das imagens
test_gen = ImageDataGenerator(rescale=1./255)
test_generation = test_gen.flow_from_directory(directory = test_directory,
                                               batch_size=4,
                                               color_mode = 'rgb',
                                               target_size =(img_height, img_width),
                                               class_mode = 'categorical',
                                               subset = None,
                                               seed = 42,
                                               interpolation='nearest')


# In[52]:


#Avaliando as imagens
evaluate = model.evaluate(test_generation, verbose=0)

print('Test loss:', evaluate[0]) 
print('Test accuracy:', evaluate[1])


# In[53]:


len(os.listdir(test_directory))


# In[54]:


print(os.listdir(os.path.join(test_directory, str(0))))


# In[55]:


#Previsões e as respostas originais
prediction = []
original = []
image = []

for i in range(len(os.listdir(test_directory))):
    for item in os.listdir(os.path.join(test_directory, str(i))):
        #print(os.listdir(os.path.join(test_directory, str(i))))
        img = cv2.imread(os.path.join(test_directory, str(i), item))
        img = cv2.resize(img, (256,256))
        image.append(img)
        img = img/255
        img = img.reshape(-1, 256, 256, 3)
        predict = model.predict(img)
        predict = np.argmax(predict)
        prediction.append(predict)
        original.append(i)


# In[56]:


print(prediction)
print(original)


# In[57]:


#Criando a Matriz de Confusão
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[58]:


#Accuracy entre as variáveis original e prediction
accuracy_score(original, prediction)


# In[59]:


#Criando uma matriz de comparação entre original e a previsão
#Gerando a figura
fig, axes = plt.subplots(5, 5, figsize=(12,12))
axes = axes.ravel()
for i in np.arange(0, 25):
    axes[i].imshow(image[i])
    axes[i].set_title('Previsão = {}\nTrue = {}'.format(str(labels_names[prediction[i]]), str(labels_names[original[i]])))
    axes[i].axis('off')
plt.subplots_adjust(wspace = 1.2)


# In[60]:


#Gerando a matriz de confusão
cm = confusion_matrix(original, prediction)

#gerando o gráfico
sns.heatmap(cm, annot=True)


# In[39]:


print(labels_names)


# In[40]:


print(classification_report(original, prediction))


# In[61]:


#Classificação de uma imagem

#Importando o pacote da biblioteca
from keras.models import load_model
#carregando o arquivo weights.hdf5
model_loaded = load_model('weights_md4.hdf5')


# In[62]:


#Visualizando a estrutura da rede neural
model_loaded.summary()


# In[63]:


#Carregando a imagem "Simulando o Front End"
img = cv2.imread(r'C:\Users\eduardo.sepulveda\IA_EXPERT\Medico\Test\2\person288_virus_587.jpeg')
#person288_virus_587
img


# In[64]:


#Dimensões da imagem
img.shape


# In[65]:


#Rederizando o tamanho da imagem
img = cv2.resize(img, (256,256))
img.shape


# In[66]:


#Normalização
img = img / 255
img


# In[67]:


#Redimensionamento
img = img.reshape(-1, 256, 256, 3)
img.shape


# In[68]:


predict = model_loaded(img)
predict


# In[69]:


predict = np.argmax(predict)
predict


# In[70]:


labels_names[predict]


# In[ ]:




