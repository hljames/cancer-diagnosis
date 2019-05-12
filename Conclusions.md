---
title: Discussion and Conclusions
notebook: Intro.ipynb
nav_include: 7
---

### Discussion of the Results

Throughout this project, we came to several important conclusions about both the dataset and the task.

1. After investigating the training data, we realized that the normal class had many images in it that shouldn't be used in training, including images that were all black, were images of text, etc. This could lead to inflated accuracy in models that use the normal set data. See Pre-Processing Raw Images and Unsupervised Clustering for example of how we handled this problem. This was also part of our motivation for training models that don't use the normal class. 

2. Our model's ability to classify the images was strongly dependent on the task, or which classes it was trying to distinguish. This explains the large discrepancy in validation accuracy between Model 0 (prediction all five classes) and Model 2 (discerning between the four abnormal classes). This might indicate that this type of data might be restricted to certain tasks. With this knowledge, care should be taken in employed these types of models in hospitals. For example, we might be capable of building a model that can discern between normal and abnormal, while the bottleneck in radiology might be in discerning between benign and malignant masses. 

3. Saliency mapping could be a reasonable method of interpreting medical imaging classification models. LIME, however, appeared to be better suited to less subtle classification tasks (at least from our exploration). Both in terms of evaluating the quality of the model and opening the "black box" to both physicians and patients, saliency maps (vanilla and guided) can be a valuable tool.

### Possible Improvements and Future Work

1. We were not able to entirely replicate the results found in our literature review -- further work could be done to more closely follow the methodology of another team in order to match their conclusions. 

2. More careful cleaning of the normal dataset could improve the results of our model. This could be done either through using a different dataset or through further work on the unsupervised clustering (truncated due to limited time). 
