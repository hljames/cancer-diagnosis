---
title: EDA
notebook: Milestone_2.ipynb
nav_include:2
---

## Contents
{:.no_toc}
*  
{: toc}

## Project Statement and EDA

<hr style="height:2pt">We have decided to exploit state-of-the-art models like ResNet and VGG through transfer learning to perform classification, as past research has demonstrated that these deep CNNs perform substantially better than hand-trained, shallower networks at classification of images from the DDSM dataset [3].

Once we have classified images, we plan to compare several possible techniques -- e.g. t-SNE, Shapley values, LICE, adversarial attack-like changes to the input -- to identify how the model classifies positive cases, especially where the model's result differs from the ground truth. We also plan to compare several approaches to help determine the amount of confidence that one can have in the classification (e.g. using a "trust score" similar to [4]), and to consider the degree to which such confidence scores help explain incorrect classifications as well as what factors make the model less certain of its classification.

Finally, we have also considered:

*   Explicitly addressing dataset bias in our model, in particular the difference in the source of "normal" images versus abnormal cases with suspicious growths, in order to prevent the classifier from making classifications based on "irrelevant" image characteristics (i.e. based on the classifier's ability to detect the dataset from which the image came). We would plan to approach this debiasing problem in a manner similar to [5].

* Conducting a user study to evaluate different methods of explanation/interpretability to physicians/radiologists/medical students

## 3. Literature review

<br>

**Transfer learning with DDSM**: We first identified which neural networks are currently being used for transfer learning in the context of cancer identification and classification [6]. We read the paper on U-net, which was trained to perform image segmentation on the neuronal structures from the ISBI challenge but can be applied to any size images. Potential challenges we would work through to do this would be developing labels, since the model classifies pixels in such a way that identifies structures, but the regions on interest in the DDSM data set are already extracted in one of the cleaner versions of the data set. We also read about ResNet, which was not specifically trained for biomedical applications, but is a widely known pre-trained model that allows much greater depth that previously possible and is widely applicable. Finally, we read about Conditional GANs, which can be used to generate captions for images that are more natural and diverse. This is a possible avenue for exploration, but it would likely require us coming up with "ground truth" captions, even if they don't especially correspond to images in the data set but rather just are more naturally structured, since a GAN discriminator must be fed both input from the generator and some "truth".

<br>

**Summary of model performance**: Jain and Levy [3] test multiple DCNN architectures on the DDSM dataset, and achieve 60.4% accuracy with a hand-built shallow CNN, 89.0% with AlexNet, and 92.9% with GoogleNet. Especially noteworthy is the fact that the GoogleNet's recall rate of 93.4% surpassed that of professional radiologists, who typically achieve recall rates that range from 74.5% to 92.3%. Shams et al. [7] perform simultaneous ROI-identification and classification using a model that combines CNNs and GANs, and achieve similar results of around 89% accuracy and 88.4% AOC on the DDSM. 

<br>

---

**Possible Methods of Interpretation/Explainability**:

- Visualization of what each neuron is detecting. Potentially label the neurons and see which are activating in a certain classification. There is existing architecture for visualizing the network, but labeling the nodes might be challenging. [Understanding Neural Networks Through Deep Visualization
22 Jun 2015 • Jason Yosinski • Jeff Clune • Anh Nguyen • Thomas Fuchs • Hod Lipson](https://paperswithcode.com/paper/understanding-neural-networks-through-deep)

- Using saliency maps / smoothgrad to visualize which pixels are most important. This would be a great first step, but it wouldn't answer questions about *why* these pixels are important, critically not answering questions like "Is it the shape of the mass? The size? The edges?"
[SmoothGrad: removing noise by adding noise
12 Jun 2017 • Daniel Smilkov • Nikhil Thorat • Been Kim • Fernanda Viégas • Martin Wattenberg](https://paperswithcode.com/paper/smoothgrad-removing-noise-by-adding-noise
)
 - An extension of this idea would be to interpret saliency maps through perterbations (like deletion, blurring, etc) [https://arxiv.org/pdf/1704.03296v3.pdf](https://arxiv.org/abs/1704.03296v3)
    -  LIME: ["Why Should I Trust You?: Explaining the Predictions of Any Classifier"
16 Feb 2016 • Marco Tulio Ribeiro • Sameer Singh • Carlos Guestrin](https://paperswithcode.com/paper/why-should-i-trust-you-explaining-the)
- BILE
- SHAPELY








## 4. Data Background

We chose to work with the USF Digital Database for Screening Mammography (DDSM), which, while widely used in the literature, requires extensive preprocessing to get to a form that can be used for even basic analysis. Given time constraints, we therefore drew on a prepared version of the dataset provided by [1] so that we could focus on the more interesting and valuable tasks of image classification and classifier interpretability. 

This version of the DDSM data differs from the original in a few ways:
- It combines the original DDSM with the CBIS-DDSM data set, which is "a subset of the DDSM data selected and curated by a trained mammographer"[2]. The CBIS-DDSM data set is cleaner and of higher resolution, but only has scans with abnormalities, so normal images were taken from the original DDSM data set and combined with this data set. That the cases with and without suspicious masses come from different data sets will have to be explicitly accounted for when training classifiers in order to ensure that the classifications are not simply made based on irrelevant features that identify the dataset (e.g. the distribution of pixel intensities), rather than the meaningful content in the image. If necessary, we may train only on abnormal observations with a growth identified, omitting the "normal" class altogether. In this case, our research question would focus on classifying growths as benign or malignant, conditional on there being a growth already identified. We have also considered the option of not omitting the "normal" observations, but instead explicitly accounting for dataset bias as in [5].
- The CBIS-DDSM images are published with associated regions of interest (ROI), identified using a pixel mask layer that varies across the images in size. The preprocessed dataset provided by [1] clips the images to the ROI using a buffer surrounding the pixel mask, though in a way that ensures the images are of uniform size. Similarly-sized tiles were also extracted from the "normal"-case images. Thus, all the images in the prepared dataset are 299 x 299 pixels.
- The CBIS-DDSM dataset contains 753 calcification cases and 891 mass cases, while the DDSM contains 695 normal cases. Given the small size of this dataset, offline data augmentation (specifically, rotations and flips) was performed to generate an enlarged sample size. However, we are concerned by the fact that it appears that augmented data were included in the test set.


The dataset is already divided into training, cross-validation, and test sets, containing 55885, 7682, and 7682 observations, respectively, though we have concatenated the cross-validation and test sets below. 

The observations are labelled as follows:
* 0: negative/normal (no suspicious masses at the time of the initial screening nor at the subsequence screening four years later) 
* 1: benign calcification
* 2: benign mass (a suspicious mass was found that was subsequently determined to be non-malignant)
* 3: malignant calcification
* 4: malignant mass

Whereas the observations of class 0 originate from the DDSM, those of classes 1-4 come from the CBIS-DDSM.

## 5. Preliminary EDA

### 5.1: Data Preparation



```python
from google.colab import drive
drive.mount("/content/gdrive")
```


    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive




```python
'''IMPORT LIBRARIES'''
import requests
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import cv2

from scipy.misc import imresize
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D
from keras.layers import Cropping2D, Conv2DTranspose, BatchNormalization, Activation
from keras.utils import np_utils, to_categorical, Sequence
from keras.losses import binary_crossentropy
from keras import backend as K,objectives
from keras.losses import mse, binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras.preprocessing import image
import tensorflow as tf
import random

from PIL import Image

from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)

os.chdir("/content/gdrive/Team Drives/AC209b_final_project/")

np.random.seed(42)
```


    Using TensorFlow backend.


The training data are stored in tfrecords files. We extract the images and store them in corresponding folder with the implemented class $\texttt{TFRecordExtractor}$. The labels and images location are stored in a $\texttt{.csv}$ file. To extract the tfrecords, we used the implementation found in [1]. Once we have post-processed the images into their respective folder, we build one global folder $\texttt{images}$ containing all the images and one $\texttt{.csv}$ file $\texttt{training_data.csv}$ containing the labels and the file locations.



```python
'''LOAD TEST AND CROSS-VALIDATION DATA'''
cv10_labels   = np.load('data/cv10_labels.npy')
cv10_data     = np.load('data/cv10_data.npy')
test10_data   = np.load('data/test10_data.npy')
test10_labels = np.load('data/test10_labels.npy')

test_data   = np.vstack((cv10_data,test10_data))
test_labels = np.concatenate((cv10_labels,test10_labels))
assert(test_data.shape[0] == len(test_labels))

test_indexes = np.array(list(range(len(test_labels))))
np.random.shuffle(test_indexes)
test_data   = test_data[test_indexes]
test_labels = test_labels[test_indexes]
```




```python
class TFRecordExtractor:
    def __init__(self, tfrecord_file, csv_name, folder_name, start_id):
        self.tfrecord_file = os.path.abspath(tfrecord_file)
        self.csv_name = csv_name
        self.folder_name = folder_name
        self.count = start_id

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        feature = {'label': tf.FixedLenFeature([], tf.int64),
                   'label_normal': tf.FixedLenFeature([], tf.int64),
                   'image': tf.FixedLenFeature([], tf.string)}

        # Decode the record read by the reader
        features = tf.parse_single_example(tfrecord, features=feature)
        
        # Convert the image data from string to numeric
        image = tf.decode_raw(features['image'], tf.uint8)

        label = features['label']
        label_normal = features['label_normal']
        image = tf.reshape(image, [299, 299, 1])
        return [image,label,label_normal]


    def post_process_images(self):
        image_data_list = self.get_images()
        b_c_df = pd.DataFrame(columns=['id', 'class', 'normal_class'])

        id_list = []
        class_list = []
        normal_class_list = []

        for image_data in image_data_list:
            self.count = self.count + 1
            file_name = self.folder_name+'//' + 'c' + str(self.count)
            id_list.append('c' + str(self.count))

            class_list.append(image_data[1])
            normal_class_list.append(image_data[2])

            cv2.imwrite(file_name+".png",image_data[0])

        id_arr = np.array(id_list)
        class_arr = np.array(class_list)
        normal_class_arr = np.array(normal_class_list)

        b_c_df["id"]  = pd.Series(id_arr)
        b_c_df["class"] = pd.Series(class_arr)
        b_c_df["normal_class"] = pd.Series(normal_class_arr)

        b_c_df.to_csv(path_or_buf=self.csv_name+'.csv', 
                      columns=['id', 'class', 'normal_class'])

    def get_images(self):
        # Initialize all tfrecord paths
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            image_data_list = []
            try:
                while True:
                    image_data = sess.run(next_element)
                    image_data_list.append(image_data)
            except:
                pass

            return image_data_list
```




```python
'''EXTRACT TRAINING DATA'''

start_id = 0
n_im_tf = 11177
t10_0 = TFRecordExtractor('/content/gdrive/Team Drives/AC209b_final_project//training10_0.tfrecords','training10_0','training10_0',start_id)
start_id += n_im_tf
t10_1 = TFRecordExtractor('/content/gdrive/Team Drives/AC209b_final_project//training10_1.tfrecords','training10_1','training10_1',start_id)
start_id += n_im_tf
t10_2 = TFRecordExtractor('/content/gdrive/Team Drives/AC209b_final_project//training10_2.tfrecords','training10_2','training10_2',start_id)
start_id += n_im_tf
t10_3 = TFRecordExtractor('/content/gdrive/Team Drives/AC209b_final_project//training10_3.tfrecords','training10_3','training10_3',start_id)
start_id += n_im_tf
t10_4 = TFRecordExtractor('/content/gdrive/Team Drives/AC209b_final_project//training10_4.tfrecords','training10_4','training10_4',start_id)

for extractor in [t10_2, t10_3, t10_4]:
    extractor.post_process_images()
```




```python
!pwd
```


    /content/gdrive/Team Drives/AC209b_final_project




```python
'''IMPORT TRAINING LABELS'''

train_label_df = pd.read_csv('data/training_data.csv')
train_label_df = train_label_df.drop(columns=['normal_class','Unnamed: 0'])
```




```python
train_label_df
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
      <th>id</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>c8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>c10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>c11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>c14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>c15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>c16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>c17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>c18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>c19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>c20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>c21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>c22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>c23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>c24</td>
      <td>4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>c25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>c26</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>c27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>c28</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>c29</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>c30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>55855</th>
      <td>c55856</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55856</th>
      <td>c55857</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55857</th>
      <td>c55858</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55858</th>
      <td>c55859</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55859</th>
      <td>c55860</td>
      <td>2</td>
    </tr>
    <tr>
      <th>55860</th>
      <td>c55861</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55861</th>
      <td>c55862</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55862</th>
      <td>c55863</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55863</th>
      <td>c55864</td>
      <td>3</td>
    </tr>
    <tr>
      <th>55864</th>
      <td>c55865</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55865</th>
      <td>c55866</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55866</th>
      <td>c55867</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55867</th>
      <td>c55868</td>
      <td>3</td>
    </tr>
    <tr>
      <th>55868</th>
      <td>c55869</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55869</th>
      <td>c55870</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55870</th>
      <td>c55871</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55871</th>
      <td>c55872</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55872</th>
      <td>c55873</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55873</th>
      <td>c55874</td>
      <td>3</td>
    </tr>
    <tr>
      <th>55874</th>
      <td>c55875</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55875</th>
      <td>c55876</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55876</th>
      <td>c55877</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55877</th>
      <td>c55878</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55878</th>
      <td>c55879</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55879</th>
      <td>c55880</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55880</th>
      <td>c55881</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55881</th>
      <td>c55882</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55882</th>
      <td>c55883</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55883</th>
      <td>c55884</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55884</th>
      <td>c55885</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>55885 rows × 2 columns</p>
</div>



### 5.2: Preliminary Analysis

**Summary tables**



```python
summary_counts_df = pd.DataFrame({'train': train_label_df['class'].value_counts(),
                                  'test': pd.Series(test_labels).value_counts()}).fillna(0)
summary_counts_df = summary_counts_df.astype(int)
summary_counts_df.index.name = 'class'
display(summary_counts_df)

totals = pd.DataFrame(summary_counts_df.sum())
totals.columns = ["totals"]
display(totals)
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
      <th>train</th>
      <th>test</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48596</td>
      <td>13360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2103</td>
      <td>558</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1911</td>
      <td>642</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1463</td>
      <td>369</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1812</td>
      <td>435</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>totals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>55885</td>
    </tr>
    <tr>
      <th>test</th>
      <td>15364</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_freq = summary_counts_df['train'].values
fig = plt.figure(figsize=(12,5))
ax0 = fig.add_subplot(121)
ax0.bar(range(0,5),class_freq,color=['k','b','g','m','r'])
ax0.set_xlabel('classes', fontsize=14)
ax0.set_ylabel('frequency', fontsize=14)
ax0.set_title('Cases per class', fontsize=16)
ax1 = fig.add_subplot(122)
ax1.bar(range(1,5),class_freq[1::],color=['b','g','m','r'])
ax1.set_xlabel('classes', fontsize=14)
ax1.set_ylabel('frequency', fontsize=14)
ax1.set_title('Cases per class (without 0)', fontsize=16)
ax1.set_xticks(range(1,5));
plt.suptitle('Class Distribution in Training Set',fontsize=25)
plt.subplots_adjust(top=0.82)

```



![png](Milestone_2_files/Milestone_2_19_0.png)


The normal type (class 0) is dominant in the dataset with the four other labels almost uniformly distributed. We will therefore need to address the imbalance between the negative (normal) and positive (i.e. suspicious-growth) classes when fitting classifiers to the training set.

**Randomly-selected instances of each class**



```python
'''FUNCTION TO PLOT IMAGE FROM ID'''
def load_image(image_id,show=False):
    '''
    -----------------------------------------------------------------------
    Function to load image from file location and normalize between 0 and 1
    -----------------------------------------------------------------------
    Param[in] image_id .. double of image id
    Param[out] path .. string of image location
    '''
    image = Image.open('./images/'+image_id+'.png')
    image = list(image.getdata())
    image = np.array(image)
    image = np.reshape(image,(299,299,1)) #We know the shape of all images are 299x299
    if show:
        plt.imshow(image[:,:,0],cmap='gray')
        plt.axis('off')
    return image/255.
```




```python
'''PLOT 5 CLASSES OF IMAGES '''

train_label_0_df = train_label_df[train_label_df['class']==0]
train_label_1_df = train_label_df[train_label_df['class']==1]
train_label_2_df = train_label_df[train_label_df['class']==2]
train_label_3_df = train_label_df[train_label_df['class']==3]
train_label_4_df = train_label_df[train_label_df['class']==4]

#Get random ids
id_0 = np.random.choice(train_label_0_df['id'])
id_1 = np.random.choice(train_label_1_df['id'])
id_2 = np.random.choice(train_label_2_df['id'])
id_3 = np.random.choice(train_label_3_df['id'])
id_4 = np.random.choice(train_label_4_df['id'])

#Plot images
fig = plt.figure(figsize=(20,20))
ax0 = fig.add_subplot(1,5,1)
load_image(id_0,True);
ax0.set_title('Class 0')
ax1 = fig.add_subplot(1,5,2)
load_image(id_1,True);
ax1.set_title('Class 1')
ax2 = fig.add_subplot(1,5,3)
load_image(id_2,True);
ax2.set_title('Class 2')
ax3 = fig.add_subplot(1,5,4)
load_image(id_3,True);
ax3.set_title('Class 3')
ax4 = fig.add_subplot(1,5,5)
load_image(id_4,True);
ax4.set_title('Class 4');
```



![png](Milestone_2_files/Milestone_2_23_0.png)


The images above are the ROI of original scans. Without medical knowledge, it seems difficult to distinguish the different classes based on the above images. However, these are random images selected from the dataset. We also plotted the mean image for each class, illustrated below, to get a sense of whether the spatial distribution of light differs systematically by class:



```python
'''PLOT MEAN IMAGE OF EACH CLASS'''
fig = plt.figure(figsize=(20,20))
axs = []
im_mean = []
for i in range(5):
    axs.append(fig.add_subplot(1,5,i+1))
    im_mean.append(np.zeros((299,299)))
    df_temp = train_label_df[train_label_df['class']==i]
    n_image_temp = df_temp.shape[0]
    for j in range(n_image_temp):
        im_temp = load_image(df_temp['id'].values[j])
        im_mean[i]+=im_temp[:,:,0]
    im_mean[i] = im_mean[i]/n_image_temp
    axs[i].imshow(im_mean[i],cmap='gray')
    axs[i].axis('off')
    axs[i].set_title('Mean image class '+str(i))
```


    0
    1
    2
    3
    4



![png](Milestone_2_files/Milestone_2_25_1.png)


Since the images were centered at the centerpoint of the growth (calcification or mass), and growths appear as light patches in the image, the average spread of the pixel/light intensity around the image center gives a general sense of whether the size of the growth differs systematically by class. 

We see that calcifications (corresponding to classes 1 and 3) are generally more "diffuse" than masses (classes 2 and 4), though that there is not a clear distinction between benign and malignant growths of either type.

**Distribution of pixel intensity by class**



```python
'''CALCULATE LIGHT AMOUNT IN EACH IMAGE'''
light_amount=[]
for i in range(5):
    light_amount.append([])
    df_temp = train_label_df[train_label_df['class']==i]
    n_image_temp = df_temp.shape[0]
    for j in range(n_image_temp):
        im_temp = load_image(df_temp['id'].values[j])
        light_amount[i].append(np.sum(im_temp))
```




```python
'''PLOT HISTOGRAMS'''
fig = plt.figure(figsize=(12,8))
plt.hist(light_amount[0],color='k',alpha=0.1,label='Class 0',density=True)
plt.hist(light_amount[1],color='b',alpha=0.1,label='Class 1',density=True)
plt.hist(light_amount[2],color='g',alpha=0.1,label='Class 2',density=True)
plt.hist(light_amount[3],color='m',alpha=0.1,label='Class 3',density=True)
plt.hist(light_amount[4],color='r',alpha=0.1,label='Class 4',density=True);
plt.xlabel('Amounth of Light per Image',fontsize=20)
plt.ylabel('Normalized frequency',fontsize=20)
plt.title('Distributions of amount of light per class',fontsize=25)
plt.legend();
```



![png](Milestone_2_files/Milestone_2_29_0.png)


From the distributions above, we see that for higher class labels, the mean of the distribution is shifted to the right and the images contain more light. The left tail of the distribution for class $0$ could be an indication of un-cleaned images with text labels or black backgrounds. 

## 6. References

[1] Eric A. Scuccimarra, DDSM dataset, Version 10. Accessed at https://github.com/escuccim/mias-mammography

[2]  Scuccimarra, Eric A. “ConvNets for Detecting Abnormalities in DDSM Mammograms.” Medium, 21 May 2018, medium.com/@ericscuccimarra/convnets-for-classifying-ddsm-mammograms-1739e0fe8028.

[3] Arzav Jain, Daniel Levy. "DeepMammo: Breast Mass Classification using Deep Convolutional Neural Networks" Accessed at http://cs231n.stanford.edu/reports/2016/pdfs/306_Report.pdf)

[4] Heinrich Jiang, Been Kim, Melody Y. Guan, Maya Gupta. "To Trust Or Not To Trust A Classifier". Accessed at https://arxiv.org/abs/1805.11783.

[5] Aditya Khosla, Tinghui Zhou, Tomasz Malisiewicz, Alexei A. Efros, Antonio Torralba. "Undoing the Damage of Dataset Bias". European Conference on Computer Vision (ECCV), 2012. Accessed at http://undoingbias.csail.mit.edu.

[6] Pengcheng Xi, Chang Shu, Rafik Goubran. "Abnormality Detection in Mammography using Deep Convolutional Neural Networks." (March 5, 2018). Accessed at https://arxiv.org/pdf/1803.01906.pdf.

[7] Shayan Shams, Richard Platania, Jian Zhang, Joohyun Kim, Kisung Lee, Seung-Jong Park. "Deep Generative Breast Cancer Screening and Diagnosis." (Sept 26, 2018). Accessed at https://link.springer.com/chapter/10.1007/978-3-030-00934-2_95.





