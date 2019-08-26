#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression


# In[2]:


images=[]
labels=[]
n=0
count=0
for i in os.listdir(f'E:/dataset/New folder/English/Fnt'):
    for j in os.listdir(f'E:/dataset/New folder/English/Fnt/{i}'):
        img=cv2.imread(f'E:/dataset/New folder/English/Fnt/{i}/{j}',cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(50,50))
        images.append(img)
        labels.append(count)
        n+=1
        if n==700:
            break
    print(n,end='')
    n=0
    count+=1
    print(count,end=" ")


# In[3]:


labels[1016-1]
#count


# In[4]:


#cv2.imshow('',images[2032])
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# In[5]:


word={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:'A',11:'B',
      12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',
      20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',
      28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',
      36:'a',37:'b',38:'c',39:'d',40:'e',41:'f',42:'g',43:'h',
      44:'i',45:'j',46:'k',47:'l',48:'m',49:'n',50:'o',51:'p',
      52:'q',53:'r',54:'s',55:'t',56:'u',57:'v',58:'w',59:'x',
      60:'y',61:'z'}


# In[6]:


X=np.array(images)
y=np.array(labels)


# In[7]:


new_x=X.reshape(len(X),-1)


# In[8]:


X.shape


# In[9]:


new_x.shape


# In[10]:


lr=LogisticRegression()


# In[ ]:





# In[11]:


lr.fit(new_x,y)


# In[13]:


test_img=cv2.imread('d:/img2.png')
test_gray=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
test_gray=cv2.resize(test_gray,(50,50))
test_X=test_gray.reshape(1,-1)
lr.predict(test_X)


# In[14]:


file=open('lr_model.pickle','wb')


# In[15]:


pickle.dump(lr,file)
file.close()


# In[24]:


labels.index(37)


# In[23]:


labels[22400]


# In[ ]:




