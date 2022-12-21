#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb


# In[2]:


img = cv2.imread('rectangles.png')


# In[3]:


plt.imshow(img)
plt.show()


# In[4]:


imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgCan = cv2.Canny(imgGray,50,200)


# In[5]:


plt.imshow(imgCan)
plt.show()


# In[6]:


blank = np.zeros_like(img)
white_bg = np.zeros((338,422,4), np.uint8)
img_list=[]
white_bg.fill(255)
img.shape


# In[7]:


i=0
angles = [29,-29,14,-14]
fig = plt.figure(figsize=(10, 7))
contour, _ = cv2.findContours(imgCan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(blank, (x, y), (x + w, y + h), (0,0,255), 2) 
    cropped_image = img[y:y+h, x:x+w]
    print([x,y,w,h])  
    i+=1
    fig.add_subplot(2, 2, i)
    plt.imshow(cropped_image)
    height, width = cropped_image.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angles[i-1], scale=1)
    print(rotate_matrix)
    rotated_image = cv2.warpAffine(src=cropped_image, M=rotate_matrix, dsize=(width, height))
    img_list.append(rotated_image)
    plt.imshow(rotated_image)


# In[8]:


plt.imshow(white_bg)


# In[9]:


len(img_list)


# In[10]:


plt.imshow(img_list[0])


# In[11]:


for i in range(len(img_list)):
    imgGray = cv2.cvtColor(img_list[i],cv2.COLOR_BGR2GRAY)
    imgCan = cv2.Canny(imgGray,50,200)
    color = [0,0,0]
    contour, _ = cv2.findContours(imgCan, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contour:
        area = cv2.contourArea(cnt)
        print(area)
        #if area<11326:
            #cv2.drawContours(blank,cnt,-1,(255,255,255),1)       
    cv2.fillPoly(img_list[i], contour, color)


# In[12]:


for i in range(len(img_list)):
    tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img_list[i])
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    dst = rgba2rgb(dst)
    img_list[i] = dst
    plt.imshow(img_list[i])


# In[13]:


white_bg = rgba2rgb(white_bg)


# In[14]:


white_bg[0:147,0:166] =img_list[0]
white_bg[150:257,10:146] = img_list[1]   
white_bg[10:73,200:326] = img_list[2]  
white_bg[150:233,200:325] = img_list[3]  


# In[15]:


img_list[0].shape


# In[16]:


plt.imshow(white_bg)


# In[ ]:




