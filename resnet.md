---
title: ResNet50 Models
notebook: resnet.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}


We prepared 6 different models with ResNet50 but using different partitions and subgroups of the version DDSM data set previously specified. The models are:
- Model 0: Raw Pre-Processed DDSM Dataset (Baseline)
    - This uses all of the data in the version of the DDSM data set that we selected.
- Model 1: Cleaned Normal Class
    - As we see in the EDA tab, this removes noisy images from the baseline DDSM data.
- Model 2: Classification of Abnormalities (Classes 1 to 4)
    - This model removes the "normal" class images and strictly classifies the different types of abnormalities. The normal images are the "cleaned" ones from Model 1.
- Model 3: Normal vs Abnormal (Class 0 vs 1-4)
    - This model reduces all the abnormal images to one class and attempts to distinguish between the normal and abnormal images.
- Model 4: Benign vs Malignant Calcification
    - This model looks strictly at the original class 1 and class 3, which are benign and malignant calcifications.
- Model 5: Benign vs Malignant Mass
    - This model looks strictly at the original class 2 and 4, which are benign and malignant masses.
    

We evaluate each model on the training data, the RGB test data, and the grayscale test data. We originally were only using the RGB test data, but because we had such low test accuracy, we worried that the Keras ImageDataGenerator converted the image to RGB in a different way than tf.image.grayscale_to_rgb did, so we saved the images as grayscale and repeated the process. Everything is in this notebook, but note this was a classic "it's the data science process!" moment where we had to take several steps back and try again.

A summary of the training and test accuracies can be found at the conclusion.



```python
'''IMPORT LIBRARIES'''
import requests
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil

from scipy.misc import imresize
from sklearn.model_selection import train_test_split

from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D
from keras.layers import Cropping2D, Conv2DTranspose, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.utils import np_utils, to_categorical
from keras.losses import binary_crossentropy
from keras import backend as K,objectives
from keras.losses import mse, binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras.preprocessing import image
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
import keras

from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)

np.random.seed(42)
```


## Loading the Data

The data was prepared into csvs that were formated to have the file name and the class labels. The code for this can be found in the EDA tab.

The validation data for training is randomly sampled from the training dataset during model creation, since that is when the data generators are created.



```python
train_df_model_0 = pd.read_csv('data/train_df_model_0.csv')
train_df_model_1 = pd.read_csv('data/train_df_model_1.csv')
train_df_model_2 = pd.read_csv('data/train_df_model_2.csv')
train_df_model_3 = pd.read_csv('data/train_df_model_3.csv')
train_df_model_4 = pd.read_csv('data/train_df_model_4.csv')
train_df_model_5 = pd.read_csv('data/train_df_model_5.csv')
```




```python
test_df_model_0 = pd.read_csv("data/test_df_model_0.csv")
test_df_model_1 = pd.read_csv("data/test_df_model_1.csv")
test_df_model_2 = pd.read_csv("data/test_df_model_2.csv")
test_df_model_3 = pd.read_csv("data/test_df_model_3.csv")
test_df_model_4 = pd.read_csv("data/test_df_model_4.csv")
test_df_model_5 = pd.read_csv("data/test_df_model_5.csv")
test_dfs = [test_df_model_0, test_df_model_1, test_df_model_2, test_df_model_3, test_df_model_4, test_df_model_5]
```


## Setting Up the Models

We will use transfer learning on the pre-trained ResNet network. Since we will train several models, the code to do so has been consolidated here.

We define several hyper-parameters for each model, and define functions to build the data generators and the model, and to evaluate them.



```python
'''HYPER-PARAMETERS'''
#Image related parameters
H = 299
W = 299
n_channels = 3

#Optimization related parameters
batch_size_train = 32
batch_size_test  = 1

#Model related parameters
model0_epochs = 5
model1_epochs = 5
model2_epochs = 15
model3_epochs = 5
model4_epochs = 15
model5_epochs = 10

model0_classes = 5
model1_classes = 5
model2_classes = 4
model3_classes = 2
model4_classes = 2
model5_classes = 2
```




```python
'''Build the model and DataGenerators.'''
def build_model(n_classes,df,x='filename',y='y', bs_train = 32, lr = 0.0001,H = H,W = W, n_channels = 3):
    #Data generator
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[y])
    val_df.reset_index(inplace=True)
    val_df.drop(['index'], axis=1, inplace=True)
    train_datagen = ImageDataGenerator(
            rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        directory='images',
        dataframe=train_df,
        x_col=x,
        y_col=y,
        # width by height only, not channels
        target_size=(H, W),
        color_mode="rgb",
        batch_size=bs_train,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        directory='images',
        dataframe=val_df,
        x_col=x,
        y_col=y,
        # width by height only, not channels
        target_size=(H, W),
        color_mode="rgb",
        batch_size=bs_train,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    #Class weight
    all_classes = df[y].unique()
    class_weights = compute_class_weight(class_weight='balanced', classes=all_classes, y=train_df[y])
    #Model architecture
    inp = Input(shape = (H,W,n_channels))

    model = ResNet50(input_shape=(H,W,n_channels), include_top=False, weights='imagenet')
    x1 = model(inp)
    x2 = GlobalAveragePooling2D()(x1)
    out = Dense(n_classes, activation='softmax')(x2)

    model = Model(inputs = inp, outputs = out)
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VAL = val_generator.n//val_generator.batch_size
        
    return model, train_generator, val_generator, class_weights, STEP_SIZE_TRAIN, STEP_SIZE_VAL
```




```python
'''Build test DataGenerators for the RGB data.'''
test_gens = {}
for i, df in zip(range(6), test_dfs):
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_dataframe(
        directory='test_images/',
        dataframe=df,
        x_col='filename',
        y_col='y',
        # width by height only, not channels
        target_size=(H, W),
        color_mode="rgb",
        batch_size=batch_size_test,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    test_gens[i] = test_generator
```


    Found 15364 images belonging to 5 classes.
    Found 15364 images belonging to 5 classes.
    Found 2004 images belonging to 4 classes.
    Found 15364 images belonging to 2 classes.
    Found 927 images belonging to 2 classes.
    Found 1077 images belonging to 2 classes.




```python
'''Build test DataGenerators for the grayscale data.'''
test_gens_gray = {}
for i, df in zip(range(6), test_dfs):
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_dataframe(
        directory='test_images_grayscale/',
        dataframe=df,
        x_col='filename',
        y_col='y',
        # width by height only, not channels
        target_size=(H, W),
        color_mode="rgb",
        batch_size=batch_size_test,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    test_gens_gray[i] = test_generator
```


    Found 15364 images belonging to 5 classes.
    Found 15364 images belonging to 5 classes.
    Found 2004 images belonging to 4 classes.
    Found 15364 images belonging to 2 classes.
    Found 927 images belonging to 2 classes.
    Found 1077 images belonging to 2 classes.




```python
'''Evaluate a given model on test and training data.'''
def evaluate_train_test(model, train_gen=None, step_size_train=None, test_gen=None):
    train_results = None
    if train_gen:
        train_results = model.evaluate_generator(train_gen, steps=step_size_train)
    test_results = None
    if test_gen:
        test_results = model.evaluate_generator(test_gen, test_gen.n)
    return train_results, test_results
```




```python
'''Format evaluation metrics from both a best weights model and a final model.'''
def pretty_metrics(model_id, model_train, model_test, model_best_weights_train, model_best_weights_test):
    if model_train and model_test and model_best_weights_train and model_best_weights_test:
        m_train_loss, m_train_acc = model_train
        m_test_loss, m_test_acc = model_test
        bw_train_loss, bw_train_acc = model_best_weights_train
        bw_test_loss, bw_test_acc = model_best_weights_test
        results = pd.DataFrame()
        results['Model'] = ['Model {}'.format(model_id), 'Best Weights Model {}'.format(model_id)]
        results['training loss'] = [m_train_loss, bw_train_loss]
        results['training acc'] = [m_train_acc, bw_train_acc]
        results['test loss'] = [m_test_loss, bw_test_loss]
        results['test acc'] = [m_test_acc, bw_test_acc]
        return results
    return None
```


## Model Training

### Model 0: Raw Pre-Processed DDSM Dataset (Baseline)

The baseline model classifies the images from the pre-processed DDSM dataset into the following $5$ classes:

- $0$: Normal
- $1$: Benign Calcification
- $2$: Benign Mass
- $3$: Malignant Calcification
- $4$: Malignant Mass



```python
model_0, train_generator_0, val_generator_0, class_weights_0, STEP_SIZE_TRAIN_0, STEP_SIZE_VAL_0 = \
    build_model(model0_classes,
                train_df_model_0,
                x='filename',
                y='y', 
                bs_train = 32, 
                lr = 0.0001,
                H = H,
                W = W,
                n_channels = 3)
```


    /usr/share/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)


    Found 44708 images belonging to 5 classes.
    Found 11177 images belonging to 5 classes.


    /usr/share/anaconda3/lib/python3.6/site-packages/keras_applications/resnet50.py:265: UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
      warnings.warn('The output shape of `ResNet50(include_top=False)` '




```python
%%time
filepath_0="models/model0_best_weights.h5"
checkpoint_0 = ModelCheckpoint(filepath_0, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_0 = [checkpoint_0]

model_0.fit_generator(generator=train_generator_0,
                    class_weight = class_weights_0,
                    steps_per_epoch=STEP_SIZE_TRAIN_0,
                    validation_data = val_generator_0,
                    validation_steps = STEP_SIZE_VAL_0,
                    epochs=model0_epochs,
                    callbacks=callbacks_list_0

)
```


    Epoch 1/5
    1397/1397 [==============================] - 2370s 2s/step - loss: 0.2246 - acc: 0.9205 - val_loss: 1.6388 - val_acc: 0.4533
    
    Epoch 00001: val_acc improved from -inf to 0.45335, saving model to models/model0_best_weights.h5
    Epoch 2/5
    1397/1397 [==============================] - 2354s 2s/step - loss: 0.1518 - acc: 0.9434 - val_loss: 2.1033 - val_acc: 0.3524
    
    Epoch 00002: val_acc did not improve from 0.45335
    Epoch 3/5
    1397/1397 [==============================] - 2352s 2s/step - loss: 0.1142 - acc: 0.9567 - val_loss: 2.6090 - val_acc: 0.4748
    
    Epoch 00003: val_acc improved from 0.45335 to 0.47483, saving model to models/model0_best_weights.h5
    Epoch 4/5
    1397/1397 [==============================] - 2352s 2s/step - loss: 0.0879 - acc: 0.9673 - val_loss: 0.5195 - val_acc: 0.8445
    
    Epoch 00004: val_acc improved from 0.47483 to 0.84450, saving model to models/model0_best_weights.h5
    Epoch 5/5
    1397/1397 [==============================] - 2350s 2s/step - loss: 0.0636 - acc: 0.9767 - val_loss: 1.8427 - val_acc: 0.5002
    
    Epoch 00005: val_acc did not improve from 0.84450
    CPU times: user 2h 32min 14s, sys: 45min 44s, total: 3h 17min 58s
    Wall time: 3h 16min 35s




```python
model_0.save('models/model_0.h5')
model_0.save_weights('models/model_0_weights.h5')
```




```python
%%time
model_0 = load_model('models/model_0.h5')
model_0_best_weights = load_model("models/model0_best_weights.h5")
```


    CPU times: user 1min 9s, sys: 1.86 s, total: 1min 11s
    Wall time: 1min 18s


#### Model 0 Evaluation

First we evaluate the models with the RGB test data.



```python
%%time
model_0_train_metrics, model_0_test_metrics = evaluate_train_test(model_0, 
                                                                  train_generator_0, 
                                                                  STEP_SIZE_TRAIN_0, 
                                                                  test_gens[0])
```


    CPU times: user 36min 20s, sys: 5min 29s, total: 41min 49s
    Wall time: 21min 36s




```python
%%time
model_0_bw_train_metrics, model_0_bw_test_metrics = evaluate_train_test(model_0_best_weights, 
                                                                        train_generator_0, 
                                                                        STEP_SIZE_TRAIN_0, 
                                                                        test_gens[0])
```


    CPU times: user 36min 22s, sys: 5min 32s, total: 41min 55s
    Wall time: 21min 45s




```python
model_0_results = pretty_metrics(0, model_0_train_metrics, 
                                 model_0_test_metrics, 
                                 model_0_bw_train_metrics, 
                                 model_0_bw_test_metrics)
display(model_0_results)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>training loss</th>
      <th>training acc</th>
      <th>test loss</th>
      <th>test acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Model 0</td>
      <td>1.741360</td>
      <td>0.534000</td>
      <td>3.356003</td>
      <td>0.407706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Best Weights Model 0</td>
      <td>0.463247</td>
      <td>0.862118</td>
      <td>1.611957</td>
      <td>0.708539</td>
    </tr>
  </tbody>
</table>
</div>


Now we evaluate the model only on the grayscale test data. We see that the results are identical to the results for the RGB test data, so we only do this as a formality for the remaining models.



```python
%%time
_ , model_0_gray_test = evaluate_train_test(model_0, test_gen=test_gens_gray[0])
_ , bw_model_0_gray_test = evaluate_train_test(model_0_best_weights, test_gen=test_gens_gray[0])
```


    CPU times: user 36min 10s, sys: 4min 16s, total: 40min 26s
    Wall time: 18min 42s




```python
pd.DataFrame({'metric': ['test loss', 'test acc'],
              'Model 0':model_0_gray_test, 
              'Model 0 Best Weights':bw_model_0_gray_test})
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric</th>
      <th>Model 0</th>
      <th>Model 0 Best Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test loss</td>
      <td>3.356003</td>
      <td>1.611957</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test acc</td>
      <td>0.407706</td>
      <td>0.708539</td>
    </tr>
  </tbody>
</table>
</div>



### Model  1: Cleaned Normal Class



```python
model_1, train_generator_1, val_generator_1, class_weights_1, STEP_SIZE_TRAIN_1, STEP_SIZE_VAL_1 = \
    build_model(model1_classes,
                train_df_model_1,
                x='filename',
                y='y', 
                bs_train = 32, 
                lr = 1.1111,
                H = H,
                W = W,
                n_channels = 3)
```


    /usr/share/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)


    Found 42261 images belonging to 5 classes.
    Found 10566 images belonging to 5 classes.


    /usr/share/anaconda3/lib/python3.6/site-packages/keras_applications/resnet50.py:265: UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
      warnings.warn('The output shape of `ResNet50(include_top=False)` '




```python
%%time
filepath_1="models/model1_best_weights.h5"
checkpoint_1 = ModelCheckpoint(filepath_1, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_1 = [checkpoint_1]

model_1.fit_generator(generator=train_generator_1,
                    class_weight = class_weights_1,
                    steps_per_epoch=STEP_SIZE_TRAIN_1,
                    validation_data = val_generator_1,
                    validation_steps = STEP_SIZE_VAL_1,
                    epochs=model1_epochs,
                    callbacks=callbacks_list_1
)
```


    Epoch 1/5
    1320/1320 [==============================] - 2145s 2s/step - loss: 2.2250 - acc: 0.8620 - val_loss: 2.2141 - val_acc: 0.8626
    
    Epoch 00001: val_acc improved from -inf to 0.86264, saving model to models/model1_best_weights.h5
    Epoch 2/5
    1320/1320 [==============================] - 2151s 2s/step - loss: 2.2245 - acc: 0.8620 - val_loss: 2.2186 - val_acc: 0.8624
    
    Epoch 00002: val_acc did not improve from 0.86264
    Epoch 3/5
    1320/1320 [==============================] - 2151s 2s/step - loss: 2.2239 - acc: 0.8620 - val_loss: 2.2248 - val_acc: 0.8620
    
    Epoch 00003: val_acc did not improve from 0.86264
    Epoch 4/5
    1320/1320 [==============================] - 2152s 2s/step - loss: 2.2241 - acc: 0.8620 - val_loss: 2.2447 - val_acc: 0.8607
    
    Epoch 00004: val_acc did not improve from 0.86264
    Epoch 5/5
    1320/1320 [==============================] - 2152s 2s/step - loss: 2.2199 - acc: 0.8623 - val_loss: 2.2232 - val_acc: 0.8621
    
    Epoch 00005: val_acc did not improve from 0.86264
    CPU times: user 1h 52min 4s, sys: 1h 9min 19s, total: 3h 1min 23s
    Wall time: 2h 59min 14s




```python
model_1.save('models/model_1.h5')
model_1.save_weights('models/model_1_weights.h5')
```




```python
model_1 = load_model('models/model_1.h5')
model_1_best_weights = load_model("models/model1_best_weights.h5")
```


#### Model Evaluation
First we evaluate the models with the RGB test data.



```python
%%time
model_1_train_metrics, model_1_test_metrics = evaluate_train_test(model_1, 
                                                                  train_generator_1, 
                                                                  STEP_SIZE_TRAIN_1, 
                                                                  test_gens[1])
```


    CPU times: user 34min 3s, sys: 5min 13s, total: 39min 17s
    Wall time: 20min 12s




```python
%%time
model_1_bw_train_metrics, model_1_bw_test_metrics = evaluate_train_test(model_1_best_weights, 
                                                                        train_generator_1, 
                                                                        STEP_SIZE_TRAIN_1, 
                                                                        test_gens[1])
```


    CPU times: user 34min 39s, sys: 5min 5s, total: 39min 45s
    Wall time: 20min 30s




```python
model_1_results = pretty_metrics(1, model_1_train_metrics, 
                                 model_1_test_metrics, 
                                 model_1_bw_train_metrics, 
                                 model_1_bw_test_metrics)
display(model_1_results)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>training loss</th>
      <th>training acc</th>
      <th>test loss</th>
      <th>test acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Model 1</td>
      <td>2.228266</td>
      <td>0.861754</td>
      <td>2.102360</td>
      <td>0.869565</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Best Weights Model 1</td>
      <td>2.222922</td>
      <td>0.862085</td>
      <td>2.101311</td>
      <td>0.869630</td>
    </tr>
  </tbody>
</table>
</div>


Now we evaluate the model only on the grayscale test data. Again, the results are identical to the RGB image results.



```python
%%time
_ , model_1_gray_test = evaluate_train_test(model_1, test_gen=test_gens_gray[1])
_ , bw_model_1_gray_test = evaluate_train_test(model_1_best_weights, test_gen=test_gens_gray[1])
```


    CPU times: user 39min 5s, sys: 4min 16s, total: 43min 22s
    Wall time: 20min 15s




```python
pd.DataFrame({'metric': ['test loss', 'test acc'],
              'Model 1':model_1_gray_test, 
              'Model 1 Best Weights':bw_model_1_gray_test})
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric</th>
      <th>Model 1</th>
      <th>Model 1 Best Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test loss</td>
      <td>2.102360</td>
      <td>2.101311</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test acc</td>
      <td>0.869565</td>
      <td>0.869630</td>
    </tr>
  </tbody>
</table>
</div>



### Model 2: Classification of Abnormalities (Classes 1 to 4)



```python
model_2, train_generator_2, val_generator_2, class_weights_2, STEP_SIZE_TRAIN_2, STEP_SIZE_VAL_2 = \
    build_model(model2_classes,
                train_df_model_2,
                x='filename',
                y='y', 
                bs_train = 32, 
                lr = 0.0001,
                H = H,
                W = W, 
                n_channels = 3)
```


    /usr/share/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)


    Found 5831 images belonging to 4 classes.
    Found 1458 images belonging to 4 classes.


    /usr/share/anaconda3/lib/python3.6/site-packages/keras_applications/resnet50.py:265: UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
      warnings.warn('The output shape of `ResNet50(include_top=False)` '




```python
%%time
filepath_2="models/model2_best_weights.h5"
checkpoint_2 = ModelCheckpoint(filepath_2, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_2 = [checkpoint_2]

model_2.fit_generator(generator=train_generator_2,
                    class_weight = class_weights_2,
                    steps_per_epoch=STEP_SIZE_TRAIN_2,
                    validation_data = val_generator_2,
                    validation_steps = STEP_SIZE_VAL_2,
                    epochs=model2_epochs,
                    callbacks=callbacks_list_2

)
```


    Epoch 1/15
    182/182 [==============================] - 330s 2s/step - loss: 1.0111 - acc: 0.5601 - val_loss: 1.3160 - val_acc: 0.3424
    
    Epoch 00001: val_acc improved from -inf to 0.34236, saving model to models/model2_best_weights.h5
    Epoch 2/15
    182/182 [==============================] - 308s 2s/step - loss: 0.6354 - acc: 0.7428 - val_loss: 1.5114 - val_acc: 0.3219
    
    Epoch 00002: val_acc did not improve from 0.34236
    Epoch 3/15
    182/182 [==============================] - 306s 2s/step - loss: 0.3922 - acc: 0.8487 - val_loss: 2.7599 - val_acc: 0.2489
    
    Epoch 00003: val_acc did not improve from 0.34236
    Epoch 4/15
    182/182 [==============================] - 306s 2s/step - loss: 0.2265 - acc: 0.9207 - val_loss: 1.8864 - val_acc: 0.3815
    
    Epoch 00004: val_acc improved from 0.34236 to 0.38149, saving model to models/model2_best_weights.h5
    Epoch 5/15
    182/182 [==============================] - 305s 2s/step - loss: 0.1558 - acc: 0.9470 - val_loss: 3.0370 - val_acc: 0.2735
    
    Epoch 00005: val_acc did not improve from 0.38149
    Epoch 6/15
    182/182 [==============================] - 305s 2s/step - loss: 0.0824 - acc: 0.9726 - val_loss: 2.7151 - val_acc: 0.3422
    
    Epoch 00006: val_acc did not improve from 0.38149
    Epoch 7/15
    182/182 [==============================] - 305s 2s/step - loss: 0.1374 - acc: 0.9528 - val_loss: 2.1769 - val_acc: 0.2994
    
    Epoch 00007: val_acc did not improve from 0.38149
    Epoch 8/15
    182/182 [==============================] - 305s 2s/step - loss: 0.1327 - acc: 0.9539 - val_loss: 2.1022 - val_acc: 0.3927
    
    Epoch 00008: val_acc improved from 0.38149 to 0.39271, saving model to models/model2_best_weights.h5
    Epoch 9/15
    182/182 [==============================] - 306s 2s/step - loss: 0.0810 - acc: 0.9723 - val_loss: 2.4061 - val_acc: 0.4327
    
    Epoch 00009: val_acc improved from 0.39271 to 0.43268, saving model to models/model2_best_weights.h5
    Epoch 10/15
    182/182 [==============================] - 306s 2s/step - loss: 0.0642 - acc: 0.9784 - val_loss: 3.2328 - val_acc: 0.3836
    
    Epoch 00010: val_acc did not improve from 0.43268
    Epoch 11/15
    182/182 [==============================] - 305s 2s/step - loss: 0.0382 - acc: 0.9883 - val_loss: 3.7338 - val_acc: 0.4158
    
    Epoch 00011: val_acc did not improve from 0.43268
    Epoch 12/15
    182/182 [==============================] - 306s 2s/step - loss: 0.0517 - acc: 0.9823 - val_loss: 2.7450 - val_acc: 0.3548
    
    Epoch 00012: val_acc did not improve from 0.43268
    Epoch 13/15
    182/182 [==============================] - 306s 2s/step - loss: 0.1056 - acc: 0.9638 - val_loss: 2.4232 - val_acc: 0.3801
    
    Epoch 00013: val_acc did not improve from 0.43268
    Epoch 14/15
    182/182 [==============================] - 305s 2s/step - loss: 0.0350 - acc: 0.9895 - val_loss: 3.0119 - val_acc: 0.3955
    
    Epoch 00014: val_acc did not improve from 0.43268
    Epoch 15/15
    182/182 [==============================] - 306s 2s/step - loss: 0.0561 - acc: 0.9809 - val_loss: 3.1049 - val_acc: 0.3478
    
    Epoch 00015: val_acc did not improve from 0.43268
    CPU times: user 1h 3s, sys: 17min 43s, total: 1h 17min 47s
    Wall time: 1h 17min 13s




```python
model_2.save('models/model_2.h5')
model_2.save_weights('models/model_2_weights.h5')
```




```python
model_2 = load_model('models/model_2.h5')
model_2_best_weights = load_model("models/model2_best_weights.h5")
```


#### Model 2 Evaluation



```python
%%time
model_2_train_metrics, model_2_test_metrics = evaluate_train_test(model_2, 
                                                                  train_generator_2, 
                                                                  STEP_SIZE_TRAIN_2, 
                                                                  test_gens[2])
```


    CPU times: user 4min 55s, sys: 42.3 s, total: 5min 37s
    Wall time: 2min 53s




```python
%%time
model_2_bw_train_metrics, model_2_bw_test_metrics = evaluate_train_test(model_2_best_weights, 
                                                                        train_generator_2, 
                                                                        STEP_SIZE_TRAIN_2, 
                                                                        test_gens[2])
```


    CPU times: user 5min 6s, sys: 43.2 s, total: 5min 49s
    Wall time: 3min 4s




```python
model_2_results = pretty_metrics(2, model_2_train_metrics, 
                                 model_2_test_metrics, 
                                 model_2_bw_train_metrics, 
                                 model_2_bw_test_metrics)
display(model_2_results)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>training loss</th>
      <th>training acc</th>
      <th>test loss</th>
      <th>test acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Model 2</td>
      <td>1.448646</td>
      <td>0.556820</td>
      <td>2.917260</td>
      <td>0.258982</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Best Weights Model 2</td>
      <td>1.583237</td>
      <td>0.550785</td>
      <td>2.279989</td>
      <td>0.265469</td>
    </tr>
  </tbody>
</table>
</div>


Now we evaluate the model only on the grayscale test data. Again, the results are identical to the RGB image results.



```python
%%time
_ , model_2_gray_test = evaluate_train_test(model_2, test_gen=test_gens_gray[2])
_ , bw_model_2_gray_test = evaluate_train_test(model_2_best_weights, test_gen=test_gens_gray[2])
```


    CPU times: user 5min 54s, sys: 33.1 s, total: 6min 27s
    Wall time: 3min 15s




```python
pd.DataFrame({'metric': ['test loss', 'test acc'],
              'Model 2':model_2_gray_test, 
              'Model 2 Best Weights':bw_model_2_gray_test})
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric</th>
      <th>Model 2</th>
      <th>Model 2 Best Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test loss</td>
      <td>2.917260</td>
      <td>2.279989</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test acc</td>
      <td>0.258982</td>
      <td>0.265469</td>
    </tr>
  </tbody>
</table>
</div>



### Model 3: Normal vs Abnormal (Class 0 vs 1-4)



```python
model_3, train_generator_3, val_generator_3, class_weights_3, STEP_SIZE_TRAIN_3, STEP_SIZE_VAL_3 = \
    build_model(model3_classes,
                train_df_model_3,
                x='filename',
                y='y', 
                bs_train = 32, 
                lr = 0.0001,
                H = H,
                W = W, 
                n_channels = 3)
```


    /usr/share/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)


    Found 42261 images belonging to 2 classes.
    Found 10566 images belonging to 2 classes.


    /usr/share/anaconda3/lib/python3.6/site-packages/keras_applications/resnet50.py:265: UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
      warnings.warn('The output shape of `ResNet50(include_top=False)` '




```python
%%time
filepath_3="models/model3_best_weights.h5"
checkpoint_3 = ModelCheckpoint(filepath_3, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_3 = [checkpoint_3]

model_3.fit_generator(generator=train_generator_3,
                    class_weight = class_weights_3,
                    steps_per_epoch=STEP_SIZE_TRAIN_3,
                    validation_data = val_generator_3,
                    validation_steps = STEP_SIZE_VAL_3,
                    epochs=model3_epochs,
                    callbacks=callbacks_list_3
)
```


    Epoch 1/5
    1320/1320 [==============================] - 2178s 2s/step - loss: 0.1021 - acc: 0.9594 - val_loss: 0.4948 - val_acc: 0.7299
    
    Epoch 00001: val_acc improved from -inf to 0.72992, saving model to models/model3_best_weights.h5
    Epoch 2/5
    1320/1320 [==============================] - 2134s 2s/step - loss: 0.0534 - acc: 0.9803 - val_loss: 1.7098 - val_acc: 0.4560
    
    Epoch 00002: val_acc did not improve from 0.72992
    Epoch 3/5
    1320/1320 [==============================] - 2129s 2s/step - loss: 0.0397 - acc: 0.9851 - val_loss: 0.1826 - val_acc: 0.9293
    
    Epoch 00003: val_acc improved from 0.72992 to 0.92928, saving model to models/model3_best_weights.h5
    Epoch 4/5
    1320/1320 [==============================] - 2130s 2s/step - loss: 0.0306 - acc: 0.9887 - val_loss: 0.5380 - val_acc: 0.6830
    
    Epoch 00004: val_acc did not improve from 0.92928
    Epoch 5/5
    1320/1320 [==============================] - 2131s 2s/step - loss: 0.0220 - acc: 0.9923 - val_loss: 0.1018 - val_acc: 0.9639
    
    Epoch 00005: val_acc improved from 0.92928 to 0.96393, saving model to models/model3_best_weights.h5
    CPU times: user 1h 55min, sys: 1h 7min 33s, total: 3h 2min 34s
    Wall time: 2h 58min 48s




```python
model_3.save('models/model_3.h5')
model_3.save_weights('models/model_3_weights.h5')
```




```python
model_3 = load_model('models/model_3.h5')
model_3_best_weights = load_model("models/model3_best_weights.h5")
```


#### Model 3 Evaluation



```python
%%time
model_3_train_metrics, model_3_test_metrics = evaluate_train_test(model_3, 
                                                                  train_generator_3, 
                                                                  STEP_SIZE_TRAIN_3, 
                                                                  test_gens[3])
```


    CPU times: user 39min 57s, sys: 4min 39s, total: 44min 37s
    Wall time: 22min 40s




```python
%%time
model_3_bw_train_metrics, model_3_bw_test_metrics = evaluate_train_test(model_3_best_weights, 
                                                                        train_generator_3, 
                                                                        STEP_SIZE_TRAIN_3, 
                                                                        test_gens[3])
```


    CPU times: user 39min 53s, sys: 4min 45s, total: 44min 38s
    Wall time: 22min 55s




```python
model_3_results = pretty_metrics(3, model_3_train_metrics, 
                                 model_3_test_metrics, 
                                 model_3_bw_train_metrics, 
                                 model_3_bw_test_metrics)
display(model_3_results)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>training loss</th>
      <th>training acc</th>
      <th>test loss</th>
      <th>test acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Model 3</td>
      <td>0.065287</td>
      <td>0.974922</td>
      <td>2.337368</td>
      <td>0.752603</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Best Weights Model 3</td>
      <td>0.065781</td>
      <td>0.974804</td>
      <td>2.338943</td>
      <td>0.752473</td>
    </tr>
  </tbody>
</table>
</div>


Now we evaluate the model only on the grayscale test data. Again, the results are identical to the RGB image results.



```python
%%time
_ , model_3_gray_test = evaluate_train_test(model_3, test_gen=test_gens_gray[3])
_ , bw_model_3_gray_test = evaluate_train_test(model_3_best_weights, test_gen=test_gens_gray[3])
```


    CPU times: user 44min 53s, sys: 4min 15s, total: 49min 9s
    Wall time: 23min 26s




```python
pd.DataFrame({'metric': ['test loss', 'test acc'],
              'Model 3':model_3_gray_test, 
              'Model 3 Best Weights':bw_model_3_gray_test})
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric</th>
      <th>Model 3</th>
      <th>Model 3 Best Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test loss</td>
      <td>2.337312</td>
      <td>2.338439</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test acc</td>
      <td>0.752669</td>
      <td>0.752603</td>
    </tr>
  </tbody>
</table>
</div>



### Model 4: Benign vs Malignant Calcification



```python
model_4, train_generator_4, val_generator_4, class_weights_4, STEP_SIZE_TRAIN_4, STEP_SIZE_VAL_4 = \
    build_model(model4_classes,
                train_df_model_4,
                x='filename',
                y='y', 
                bs_train = 32, 
                lr = 0.0001,
                H = H,
                W = W, 
                n_channels = 3)
```


    /usr/share/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)


    Found 2852 images belonging to 2 classes.
    Found 714 images belonging to 2 classes.


    /usr/share/anaconda3/lib/python3.6/site-packages/keras_applications/resnet50.py:265: UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
      warnings.warn('The output shape of `ResNet50(include_top=False)` '




```python
%%time
filepath_4="models/model4_best_weights.h5"
checkpoint_4 = ModelCheckpoint(filepath_4, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_4 = [checkpoint_4]

model_4.fit_generator(generator=train_generator_4,
                    class_weight = class_weights_4,
                    steps_per_epoch=STEP_SIZE_TRAIN_4,
                    validation_data = val_generator_4,
                    validation_steps = STEP_SIZE_VAL_4,
                    epochs=model4_epochs,
                    callbacks=callbacks_list_4
)
```


    Epoch 1/15
    89/89 [==============================] - 195s 2s/step - loss: 0.6529 - acc: 0.6310 - val_loss: 1.4264 - val_acc: 0.6151
    
    Epoch 00001: val_acc improved from -inf to 0.61506, saving model to models/model4_best_weights.h5
    Epoch 2/15
    89/89 [==============================] - 149s 2s/step - loss: 0.4623 - acc: 0.7749 - val_loss: 1.0198 - val_acc: 0.5777
    
    Epoch 00002: val_acc did not improve from 0.61506
    Epoch 3/15
    89/89 [==============================] - 149s 2s/step - loss: 0.2559 - acc: 0.8975 - val_loss: 1.3097 - val_acc: 0.4868
    
    Epoch 00003: val_acc did not improve from 0.61506
    Epoch 4/15
    89/89 [==============================] - 149s 2s/step - loss: 0.2087 - acc: 0.9150 - val_loss: 1.3952 - val_acc: 0.6232
    
    Epoch 00004: val_acc improved from 0.61506 to 0.62317, saving model to models/model4_best_weights.h5
    Epoch 5/15
    89/89 [==============================] - 149s 2s/step - loss: 0.1237 - acc: 0.9561 - val_loss: 2.5044 - val_acc: 0.6232
    
    Epoch 00005: val_acc improved from 0.62317 to 0.62317, saving model to models/model4_best_weights.h5
    Epoch 6/15
    89/89 [==============================] - 149s 2s/step - loss: 0.1640 - acc: 0.9414 - val_loss: 3.9844 - val_acc: 0.5792
    
    Epoch 00006: val_acc did not improve from 0.62317
    Epoch 7/15
    89/89 [==============================] - 149s 2s/step - loss: 0.1075 - acc: 0.9596 - val_loss: 0.9756 - val_acc: 0.6540
    
    Epoch 00007: val_acc improved from 0.62317 to 0.65396, saving model to models/model4_best_weights.h5
    Epoch 8/15
    89/89 [==============================] - 149s 2s/step - loss: 0.0559 - acc: 0.9831 - val_loss: 1.1602 - val_acc: 0.7038
    
    Epoch 00008: val_acc improved from 0.65396 to 0.70381, saving model to models/model4_best_weights.h5
    Epoch 9/15
    89/89 [==============================] - 149s 2s/step - loss: 0.0702 - acc: 0.9789 - val_loss: 1.1903 - val_acc: 0.6613
    
    Epoch 00009: val_acc did not improve from 0.70381
    Epoch 10/15
    89/89 [==============================] - 149s 2s/step - loss: 0.0586 - acc: 0.9803 - val_loss: 1.3737 - val_acc: 0.6173
    
    Epoch 00010: val_acc did not improve from 0.70381
    Epoch 11/15
    89/89 [==============================] - 149s 2s/step - loss: 0.0581 - acc: 0.9782 - val_loss: 2.6281 - val_acc: 0.5836
    
    Epoch 00011: val_acc did not improve from 0.70381
    Epoch 12/15
    89/89 [==============================] - 149s 2s/step - loss: 0.0616 - acc: 0.9793 - val_loss: 0.9857 - val_acc: 0.6584
    
    Epoch 00012: val_acc did not improve from 0.70381
    Epoch 13/15
    89/89 [==============================] - 149s 2s/step - loss: 0.0312 - acc: 0.9898 - val_loss: 0.9508 - val_acc: 0.7097
    
    Epoch 00013: val_acc improved from 0.70381 to 0.70968, saving model to models/model4_best_weights.h5
    Epoch 14/15
    89/89 [==============================] - 150s 2s/step - loss: 0.0257 - acc: 0.9905 - val_loss: 1.1822 - val_acc: 0.6833
    
    Epoch 00014: val_acc did not improve from 0.70968
    Epoch 15/15
    89/89 [==============================] - 149s 2s/step - loss: 0.0421 - acc: 0.9856 - val_loss: 1.0439 - val_acc: 0.5968
    
    Epoch 00015: val_acc did not improve from 0.70968
    CPU times: user 30min 20s, sys: 8min 22s, total: 38min 43s
    Wall time: 38min 37s




```python
model_4.save('models/model_4.h5')
model_4.save_weights('models/model_4_weights.h5')

```




```python
model_4 = load_model('models/model_4.h5')
model_4_best_weights = load_model("models/model4_best_weights.h5")
```


#### Model 4 Evaluation



```python
%%time
model_4_train_metrics, model_4_test_metrics = evaluate_train_test(model_4, 
                                                                  train_generator_4, 
                                                                  STEP_SIZE_TRAIN_4, 
                                                                  test_gens[4])
```


    CPU times: user 2min 33s, sys: 19.3 s, total: 2min 52s
    Wall time: 1min 28s




```python
%%time
model_4_bw_train_metrics, model_4_bw_test_metrics = evaluate_train_test(model_4_best_weights, 
                                                                        train_generator_4, 
                                                                        STEP_SIZE_TRAIN_4, 
                                                                        test_gens[4])
```


    CPU times: user 2min 51s, sys: 20 s, total: 3min 11s
    Wall time: 1min 48s




```python
model_4_results = pretty_metrics(4, model_4_train_metrics, 
                                 model_4_test_metrics, 
                                 model_4_bw_train_metrics, 
                                 model_4_bw_test_metrics)
display(model_4_results)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>training loss</th>
      <th>training acc</th>
      <th>test loss</th>
      <th>test acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Model 4</td>
      <td>0.885222</td>
      <td>0.637234</td>
      <td>1.321746</td>
      <td>0.464941</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Best Weights Model 4</td>
      <td>0.477532</td>
      <td>0.813121</td>
      <td>1.709040</td>
      <td>0.521036</td>
    </tr>
  </tbody>
</table>
</div>


Now we evaluate the model only on the grayscale test data. Again, the results are identical to the RGB image results.



```python
%%time
_ , model_4_gray_test = evaluate_train_test(model_4, test_gen=test_gens_gray[4])
_ , bw_model_4_gray_test = evaluate_train_test(model_4_best_weights, test_gen=test_gens_gray[4])
```


    CPU times: user 3min 56s, sys: 15.3 s, total: 4min 11s
    Wall time: 2min 34s




```python
pd.DataFrame({'metric': ['test loss', 'test acc'],
              'Model 4':model_4_gray_test, 
              'Model 4 Best Weights':bw_model_4_gray_test})
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric</th>
      <th>Model 4</th>
      <th>Model 4 Best Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test loss</td>
      <td>1.321746</td>
      <td>1.709040</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test acc</td>
      <td>0.464941</td>
      <td>0.521036</td>
    </tr>
  </tbody>
</table>
</div>



### Model 5: Benign vs Malignant Mass



```python
model_5, train_generator_5, val_generator_5, class_weights_5, STEP_SIZE_TRAIN_5, STEP_SIZE_VAL_5 = \
    build_model(model5_classes,
                train_df_model_5,
                x='filename',
                y='y', 
                bs_train = 32, 
                lr = 0.0001,
                H = H,
                W = W, 
                n_channels = 3)
```


    /usr/share/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)


    Found 2978 images belonging to 2 classes.
    Found 745 images belonging to 2 classes.


    /usr/share/anaconda3/lib/python3.6/site-packages/keras_applications/resnet50.py:265: UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
      warnings.warn('The output shape of `ResNet50(include_top=False)` '




```python
%%time
filepath_5="models/model5_best_weights.h5"
checkpoint_5 = ModelCheckpoint(filepath_5, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_5 = [checkpoint_5]

model_5.fit_generator(generator=train_generator_5,
                    class_weight = class_weights_5,
                    steps_per_epoch=STEP_SIZE_TRAIN_5,
                    validation_data = val_generator_5,
                    validation_steps = STEP_SIZE_VAL_5,
                    epochs=model5_epochs,
                    callbacks=callbacks_list_5

)
```


    Epoch 1/10
    93/93 [==============================] - 221s 2s/step - loss: 0.6168 - acc: 0.6781 - val_loss: 0.8356 - val_acc: 0.5163
    
    Epoch 00001: val_acc improved from -inf to 0.51630, saving model to models/model5_best_weights.h5
    Epoch 2/10
    93/93 [==============================] - 157s 2s/step - loss: 0.4287 - acc: 0.8054 - val_loss: 0.6554 - val_acc: 0.6480
    
    Epoch 00002: val_acc improved from 0.51630 to 0.64797, saving model to models/model5_best_weights.h5
    Epoch 3/10
    93/93 [==============================] - 156s 2s/step - loss: 0.2297 - acc: 0.9133 - val_loss: 0.9277 - val_acc: 0.6886
    
    Epoch 00003: val_acc improved from 0.64797 to 0.68864, saving model to models/model5_best_weights.h5
    Epoch 4/10
    93/93 [==============================] - 156s 2s/step - loss: 0.1423 - acc: 0.9462 - val_loss: 0.8214 - val_acc: 0.6760
    
    Epoch 00004: val_acc did not improve from 0.68864
    Epoch 5/10
    93/93 [==============================] - 156s 2s/step - loss: 0.0890 - acc: 0.9731 - val_loss: 0.6199 - val_acc: 0.7195
    
    Epoch 00005: val_acc improved from 0.68864 to 0.71950, saving model to models/model5_best_weights.h5
    Epoch 6/10
    93/93 [==============================] - 156s 2s/step - loss: 0.0872 - acc: 0.9677 - val_loss: 0.7425 - val_acc: 0.6732
    
    Epoch 00006: val_acc did not improve from 0.71950
    Epoch 7/10
    93/93 [==============================] - 156s 2s/step - loss: 0.0695 - acc: 0.9745 - val_loss: 0.6562 - val_acc: 0.6662
    
    Epoch 00007: val_acc did not improve from 0.71950
    Epoch 8/10
    93/93 [==============================] - 156s 2s/step - loss: 0.1780 - acc: 0.9315 - val_loss: 0.8216 - val_acc: 0.6073
    
    Epoch 00008: val_acc did not improve from 0.71950
    Epoch 9/10
    93/93 [==============================] - 156s 2s/step - loss: 0.0510 - acc: 0.9805 - val_loss: 0.6686 - val_acc: 0.6606
    
    Epoch 00009: val_acc did not improve from 0.71950
    Epoch 10/10
    93/93 [==============================] - 156s 2s/step - loss: 0.0297 - acc: 0.9886 - val_loss: 0.8380 - val_acc: 0.7377
    
    Epoch 00010: val_acc improved from 0.71950 to 0.73773, saving model to models/model5_best_weights.h5
    CPU times: user 21min 52s, sys: 5min 50s, total: 27min 43s
    Wall time: 27min 37s




```python
model_5.save('models/model_5.h5')
model_5.save_weights('models/model_5_weights.h5')
```




```python
filepath_5="models/model5_best_weights.h5"
model_5 = load_model('models/model_5.h5')
model_5_best_weights = load_model(filepath_5)
```


#### Model 5 Evaluation



```python
%%time
model_5_train_metrics, model_5_test_metrics = evaluate_train_test(model_5, 
                                                                  train_generator_5, 
                                                                  STEP_SIZE_TRAIN_5, 
                                                                  test_gens[5])
```


    CPU times: user 2min 57s, sys: 20.2 s, total: 3min 17s
    Wall time: 1min 41s




```python
%%time
model_5_bw_train_metrics, model_5_bw_test_metrics = evaluate_train_test(model_5_best_weights, 
                                                                        train_generator_5, 
                                                                        STEP_SIZE_TRAIN_5, 
                                                                        test_gens[5])
```


    CPU times: user 3min 25s, sys: 21.2 s, total: 3min 46s
    Wall time: 2min 10s




```python
model_5_results = pretty_metrics(5, model_5_train_metrics, 
                                 model_5_test_metrics, 
                                 model_5_bw_train_metrics, 
                                 model_5_bw_test_metrics)
display(model_5_results)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>training loss</th>
      <th>training acc</th>
      <th>test loss</th>
      <th>test acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Model 5</td>
      <td>0.274897</td>
      <td>0.897828</td>
      <td>1.513099</td>
      <td>0.517177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Best Weights Model 5</td>
      <td>0.272092</td>
      <td>0.899185</td>
      <td>1.513085</td>
      <td>0.517177</td>
    </tr>
  </tbody>
</table>
</div>


Now we evaluate the model only on the grayscale test data. Again, the results are identical to the RGB image results.



```python
%%time
_ , model_5_gray_test = evaluate_train_test(model_5, test_gen=test_gens_gray[5])
_ , bw_model_5_gray_test = evaluate_train_test(model_5_best_weights, test_gen=test_gens_gray[5])
```


    CPU times: user 5min 7s, sys: 17.4 s, total: 5min 25s
    Wall time: 3min 23s




```python
pd.DataFrame({'metric': ['test loss', 'test acc'],
              'Model 5':model_5_gray_test, 
              'Model 5 Best Weights':bw_model_5_gray_test})
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric</th>
      <th>Model 5</th>
      <th>Model 5 Best Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test loss</td>
      <td>1.513099</td>
      <td>1.513085</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test acc</td>
      <td>0.517177</td>
      <td>0.517177</td>
    </tr>
  </tbody>
</table>
</div>



## Summary of Results

because the RGB and grayscale test data results were essentially the same, we simply show the test results from the RGB data.



```python
all_results = pd.concat([model_0_results, 
                         model_1_results, model_2_results, model_3_results, model_4_results, model_5_results])
display(all_results_df)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>training loss</th>
      <th>training acc</th>
      <th>test loss</th>
      <th>test acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Model 0</td>
      <td>1.74136</td>
      <td>0.534</td>
      <td>3.356003</td>
      <td>0.407706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Best Weights Model 0</td>
      <td>0.463247</td>
      <td>0.862118</td>
      <td>1.611957</td>
      <td>0.708539</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Model 1</td>
      <td>2.228266</td>
      <td>0.861754</td>
      <td>2.10236</td>
      <td>0.869565</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Model 1 Best Weights</td>
      <td>2.222922</td>
      <td>0.862085</td>
      <td>2.101311</td>
      <td>0.86963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Model 2</td>
      <td>1.448646</td>
      <td>0.55682</td>
      <td>2.91726</td>
      <td>0.258982</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Best Weights Model 2</td>
      <td>1.583237</td>
      <td>0.550785</td>
      <td>2.279989</td>
      <td>0.265469</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Model 3</td>
      <td>0.065287</td>
      <td>0.974922</td>
      <td>2.337368</td>
      <td>0.752603</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Best Weights Model 3</td>
      <td>0.065781</td>
      <td>0.974804</td>
      <td>2.338943</td>
      <td>0.752473</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Model 4</td>
      <td>0.885222</td>
      <td>0.637234</td>
      <td>1.321746</td>
      <td>0.464941</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Best Weights Model 4</td>
      <td>0.477532</td>
      <td>0.813121</td>
      <td>1.70904</td>
      <td>0.521036</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Model 5</td>
      <td>0.274897</td>
      <td>0.897828</td>
      <td>1.513099</td>
      <td>0.517177</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Best Weights Model 5</td>
      <td>0.272092</td>
      <td>0.899185</td>
      <td>1.513085</td>
      <td>0.517177</td>
    </tr>
  </tbody>
</table>
</div>


## Conclusion

There results were not ideal because the accuracies are not comparable to what we've seen in the literature review. The fact that the validation accuracy and the test accuracy are not especially close is worrisome especially when the trianing data accuracy was so high durin training. Based on these results, we decided to take a step back and consider other models.
