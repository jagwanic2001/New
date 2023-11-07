#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout


# In[9]:


from sklearn.datasets import load_digits
mnist = load_digits()


# In[10]:


pip install keras


# In[11]:


import keras


# In[12]:


from keras.datasets import mnist


# In[13]:


(X_train,y_train),(X_test,y_test)=mnist.load_data()


# In[14]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[15]:


def plot_input_img(i):
    plt.imshow(X_train[i],cmap='binary')
    plt.title(y_train[i])
    plt.show()


# In[16]:


for i in range(10):
    plot_input_img(i)


# In[17]:


X_train=X_train.astype(np.float32)/255
X_test=X_test.astype(np.float32)/255


# In[18]:


X_train=np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)


# In[19]:


y_train = keras.utils.to_categorical(y_train)
y_test= keras.utils.to_categorical(y_test)
    


# In[20]:


y_train


# In[21]:


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = (28,28,1),activation ="relu" ))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10,activation = "softmax"))


# In[22]:


model.summary()


# In[23]:


model.compile(optimizer='adam',loss = keras.losses.categorical_crossentropy,metrics = ['accuracy'])


# In[24]:


from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[25]:


es = EarlyStopping(monitor='val_acc',min_delta=0.1,patience = 4,verbose = 1)


# In[26]:


mc = ModelCheckpoint("./bestmodel.h5",monitor = "val_acc",verbose = 1,save_best_only = True)


# In[27]:


cb=[es,mc]


# In[ ]:


his = model.fit(X_train,y_train,epochs = 50,validation_split=0.3,callbacks = cb)


# In[ ]:


his = model.fit(X_train,y_train,batch_size = 128,epochs = 50,verbose=1,validation_data = (X_test,y_test))


# In[21]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[23]:


score = model.evaluate(X_test,y_test)
print(f"the model accuracy is {score[1]} ")


# In[9]:


import pygame,sys
from pygame import image
from pygame.locals import*
from keras.models import load_model
from tokenize import Number
from numpy.lib.type_check import imag
from numpy import testing
from tensorflow.python.keras.backend import constant


import cv2
import numpy as np


# In[10]:



WHITE = (255,255,255 )
BLACK = (255,255,255)
RED = (255,0,0) 
WINDOWSIZEX= 640
WINDOWSIZEY=480 
BOUNDRYINC = 5
IMAGESAVE = False



MODEL = load_model("bestmodel.h5")
PREDICT = True
LABELS = {0:"Zero",1:"One",
          2:"Two",3:"Three",
          4:"Four",5:"Five",
          6:"six",7:"seven",
         8:"eight",9:"nine"}
pygame.init()
DISPLAYSURF=pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))
pygame.display.set_caption("Digit Board")
iswriting = False
number_xcord=[]
number_ycord = []
image_cnt = 1
REDICT = True
FONT=pygame.font.Font("freesansbold.ttf",18)
x=10
y=10

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord,ycord=event.pos
            pygame.draw.circle(DISPLAYSURF,WHITE,(xcord,ycord),4,0)
            
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            
            
            rect_min_x,rect_max_x = max(number_xcord[0]-BOUNDRYINC,0 ),min(WINDOWSIZEX,number_xcord[-1]+BOUNDRYINC)
            rect_min_y,rect_max_y = max(number_ycord[0]-BOUNDRYINC ,0),min(number_ycord[-1]+BOUNDRYINC,WINDOWSIZEX)
            number_xcord = []
            number_ycord = []
            ing_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
            if IMAGESAVE:
                cv2.imwrite('image.png')
                image_cnt+=1
            if PREDICT:
                image = cv2.resize(ing_arr,(28,28))
                image = np.pad(image,(10,10),'constant',constant_values=0)
                image = cv2.resize(image,(28,28))/255
                
                
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
                textSurface = FONT.render(label,True,RED,WHITE)
                textRecobj = textSurface.get_rect()
                textRecobj.left ,textRecobj.bottom=rect_min_x,rect_max_y
                DISPLAYSURF.blit(textSurface,textRecobj)
                
                
                
            if event.type == KEYDOWN:
                if event.unicode =="n":
                    DISPLAYSURF.fill(BLACK)
    pygame.display.update()

    
            


# import cv2

# In[37]:





# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:





# In[ ]:





# In[5]:





# In[ ]:





# In[2]:





# In[23]:





# In[24]:





# In[25]:





# In[26]:





# In[32]:





# In[ ]:





# In[ ]:





# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




