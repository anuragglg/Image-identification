#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


pip install opencv-python


# In[3]:


import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


# In[4]:


img=image.load_img(r'F:\ANURAG GUPTA\cell_images2\val\SUBCLINICAL\IR_11849.jpg')


# In[5]:


img.show()


# In[6]:


plt.imshow(img)


# In[7]:


cv2.imread(r'F:\ANURAG GUPTA\cell_images2\val\SUBCLINICAL\IR_11849.jpg')


# In[8]:


cv2.imread(r'F:\ANURAG GUPTA\cell_images2\val\SUBCLINICAL\IR_11849.jpg').shape


# In[9]:


train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)


# In[10]:


train_dataset=train.flow_from_directory(r'F:\ANURAG GUPTA\Clinical vs normal\train',
                                      target_size=(200,200),
                                      batch_size=100,
                                      class_mode='binary')

validation_dataset=validation.flow_from_directory(r'F:\ANURAG GUPTA\Clinical vs normal\val',
                                      target_size=(200,200),
                                      batch_size=20,
                                      class_mode='binary')


# In[11]:


train_dataset.class_indices


# In[12]:


train_dataset.classes


# In[13]:


model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                 tf.keras.layers.MaxPool2D(2,2),
                                  #
                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  #
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  ##
                                  tf.keras.layers.Flatten(),
                                  ##
                                  tf.keras.layers.Dense(512,activation='relu'),
                                  ##
                                  tf.keras.layers.Dense(1,activation='sigmoid')
                                  ])


# In[14]:


model.compile(loss='binary_crossentropy',
             optimizer= RMSprop(learning_rate=0.001),
             metrics=['accuracy'])


# In[15]:


model_fit = model.fit(train_dataset,
                   steps_per_epoch = 10,
                   epochs= 30,
                   validation_data= validation_dataset)


# In[16]:


dir_path = r'F:\ANURAG GUPTA\Sub-clinical vs normal\test\SUBCLINICAL'

for i in os.listdir(dir_path ):
    img = image.load_img(dir_path + "//" + i, target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    X= image.img_to_array(img)
    X= np.expand_dims(X,axis=0)
    images=np.vstack([X])
    
    val = model.predict(images)
    if val == 0:
        print("Clinical" +str(val))
    elif val==1:
        print("Normal"+str(val))
    else:
        print(val)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




