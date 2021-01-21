#%%
!pip install livelossplot
#%%
from livelossplot.tf_keras import PlotLossesCallback
import numpy as np
import tensorflow as tf
import warnings;warnings.simplefilter('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
np.random.seed(42)
%matplotlib inline
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
# %%
train_images = pd.read_csv('dataset/train/images.csv',header=None)
train_labels = pd.read_csv('dataset/train/labels.csv',header=None)
val_images = pd.read_csv('dataset/validation/images.csv',header=None)
val_labels = pd.read_csv('dataset/validation/labels.csv',header=None)
# %%
train_images.head()
train_labels.head()
# %%
print(train_images.shape,train_labels.shape)
print(val_images.shape,val_labels.shape)
# %%
X_train = train_images.values.reshape(3200,64,128,1)
X_val = val_images.values.reshape(800,64,128,1)
y_train = train_labels.values
y_val = val_labels.values
print(X_train,X_val)
print(y_train,y_val)
# %%
plt.figure(0,figsize=(12,12))
for i in range(1,4):
    plt.subplot(1,3,i)
    img = np.squeeze(X_train[np.random.randint(0,X_train.shape[0])])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
plt.imshow(np.squeeze(X_train[3]),cmap='gray')
# %%
datagen_train = ImageDataGenerator(horizontal_flip=True)
datagen_train.fit(X_train)
datagen_val = ImageDataGenerator(horizontal_flip=True)
datagen_val.fit(X_val)
#%%
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
# %%
model = tf.keras.Sequential()

model.add(Conv2D(32,(5,5),padding='same',input_shape=(64,128,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model = Sequential()
model.add(Conv2D(64,(5,5),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Dense(4,activation='softmax'))
# %%
initial_learning_rate = 0.005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps=5,
                                                            decay_rate=0.96,
                                                            staircase=True)
optimizer = Adam(learning_rate=lr_schedule)
# %%
model.compile(optimizer = optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

# %%
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_loss',
                             save_weights_only=True, mode='min', verbose=0)
callbacks = [PlotLossesCallback(), checkpoint]
batch_size = 32
history = model.fit(
    datagen_train.flow(X_train,y_train,batch_size = batch_size,shuffle = True),
    steps_per_epoch=len(X_train)//batch_size,
    validation_data=datagen_val.flow(X_val,y_val,batch_size=batch_size,shuffle=True),
    validation_steps=len(X_val)//batch_size,
    epochs=12,
    callbacks = callbacks
    )
# %%
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
model.evaluate(X_val,y_val)
y_true = np.argmax(model.predict(X_val),1)
print(metrics.classification_report(y_true,y_pred))
print(%metrics.accuracy_score(y_true,y_pred))

labels = ["squiggle", "narrowband", "noise", "narrowbanddrd"]

ax= plt.subplot()
sns.heatmap(metrics.confusion_matrix(y_true, y_pred, normalize='true'), annot=True, ax = ax, cmap=plt.cm.Blues); #annot=True to annotate cells

# labels, title and ticks
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)