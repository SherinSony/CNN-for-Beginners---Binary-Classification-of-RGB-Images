
# coding: utf-8

# RGB IMAGE CLASSIFICATION USING KERAS

# import the necessary packages
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input, merge,concatenate
from keras.models import Model
from keras.models import Sequential
import pandas as pd
from keras.models import load_model
from keras import backend as K
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

# initialize the parameters
EPOCHS = 25
INIT_LR = 1e-3
BS = 50

# image paths and randomly shuffle images
dataset="D:\DL\image-classification-keras\image-classification-keras\images"
imagePaths = sorted(list(paths.list_images(dataset)))

random.seed(42)
random.shuffle(imagePaths) #mix the images from the 2 folders

# loop over the input images
data = []
labels = []
for imagePath in imagePaths:
    
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)  
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "santa" else 0
    labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)


# partition the data into training and testing splits 
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.20, random_state=42)

testY = to_categorical(testY, num_classes=2) #num_classes may give issue for >2 classes, better not to use this feature

# convert into a matrix with one-hot-encoded values
trainY = to_categorical(trainY, num_classes=2)


# --------------- The NN layers  ---------------------

# initialize the model
model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3,3), strides=1, input_shape=(28,28,3),name="Layer1Conv2D") ) #change shape of images
model.add(BatchNormalization(axis=-1,name="Layer2BN"))  #-1 for channels_last
model.add(Activation('relu',name="Layer3ActRelu"))

model.add(Conv2D(filters=16, kernel_size=(3,3), strides=1,name="Layer4Conv2D"))
model.add(BatchNormalization(axis=-1,name="Layer5BN"))
model.add(Activation('relu',name="Layer6ActRelu"))
          
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,name="Layer7Conv2D" ))
model.add(BatchNormalization(axis=-1,name="Layer8BN"))
model.add(Activation('relu',name="Layer9ActRelu"))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=2,name="Layer10Conv2D" ))
model.add(BatchNormalization(axis=-1,name="Layer11BN"))
model.add(Activation('relu',name="Layer12ActRelu"))

model.add(Flatten(name="Layer13Flat"))
model.add(Dense(100, activation='relu',name="Layer14DenseActRelu"))
model.add(Dropout(.5,name="Layer15DropOut"))
model.add(Dense(2, activation='softmax',name="Layer16DenseSoftmax")) # 2 for Binary classification

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) # note the loss function names

# see how it looks like and check how many parameters you have - play around with the layers above and see the changes here!
model.summary(line_length=70)

# train the network
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, vertical_flip=True,shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1)


# plot the training loss and accuracy
plt.style.use("classic")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")  #upper or top
plt.show()
plt.savefig("D:\DL\model1_plot")


# ------ Test on new images--------
#Loop through the Test images, capture the predictions in 2 lists and file name in another and merge all 3 into 1 dataframe

Val_dataset="D:\DL\image-classification-keras\image-classification-keras\Val_examples"
Val_imagePaths = sorted(list(paths.list_images(Val_dataset)))

#you can use a single data frame with many cols also instead
myfile1=[]
myfile2=[]
myfile3=[]

for imagePath in Val_imagePaths:
    #print(imagePath)
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28)) #change the size
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    (notSanta, santa) = model.predict_proba(image)[0]
    myfile1.append(notSanta)
    myfile2.append(santa)
    myfile3.append(imagePath)
    
    label = "Santa" if santa > notSanta else "Not Santa"
    proba = santa if santa > notSanta else notSanta
    label = "{}: {:.2f}%".format(label, proba * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)  #each image will pop out , close the image to stop this command - this command will show the image actually
    

df=pd.DataFrame(
    {"Not Santa":myfile1,
     "Santa":myfile2,
     "PathOfImage":myfile3
    }
)
df

df.to_csv("D:\DL\image-classification-keras\image-classification-keras\Test results_3.csv")

# save the model to disk
model.save("D:\DL\model3.hdf5")




