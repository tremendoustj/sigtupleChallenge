import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.misc import imsave

train_dir = "C:\\Users\\tjain\\Downloads\\SigTuple_data\\Train_Data\\"
#training_data = []
trainData = []
responses = []
if not os.path.isdir(train_dir):
      raise IOError("The folder " + train_dir + " doesn't exist")
for root, dirs, files in os.walk(train_dir):
    for filename in (x for x in files if x.endswith('.jpg')):
        filepath = os.path.join(root, filename)
        # print(filepath)
        #object_class = filepath.split('\\')[-2]
        #training_data.append({'object_class': object_class,
        #                       'image_path': filepath})
        if filepath.endswith('-mask.jpg'):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            # ("Masked Image Size = ",len(img))
            img = img.ravel()
               # np.set_printoptions(threshold=np.nan)
            for x in img:
                responses.append(x)
        else:
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                # np.set_printoptions(threshold=np.nan)
                # print("image size = ",len(img))
            #img = img.ravel()
                # img = img.flatten()
            for x in img:
                for y in x:
                    trainData.append(y)

trainData = np.array(trainData)
responses = np.array(responses)
trainData = np.reshape(trainData,(-1,3)).astype(np.float32)
print(np.shape(trainData))
responses= np.reshape(responses,(-1,1)).astype(np.float32)

knn = cv2.ml.KNearest_create()

print(np.shape(trainData))
print(np.shape(responses))

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE, responses)


test_dir = "C:\\Users\\tjain\\Downloads\\SigTuple_data\\Test_Data\\D0F6DE661D64.jpg"

newcomer = cv2.imread(test_dir, cv2.IMREAD_COLOR)
            # ("Masked Image Size = ",len(img))
print(np.shape(newcomer))
h,w = np.shape(newcomer)[0], np.shape(newcomer)[1]
newcomer = newcomer.reshape(-1,3).astype(np.float32)
print("Finding Nearest Neighbour")
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)

results = results.reshape(h,w).astype(np.uint8)
print(results)

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((15,15),np.uint8)

closing = cv2.morphologyEx(results, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)

imsave(test_dir[:-4]+"-mask"+test_dir[-4:],opening)

