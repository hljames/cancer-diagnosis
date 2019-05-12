---
title: Introduction
notebook: Intro.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}

## Literature review

**Transfer learning with DDSM**: We first identified which neural networks are currently being used for transfer learning in the context of cancer identification and classification [6]. We read the paper on U-net, which was trained to perform image segmentation on the neuronal structures from the ISBI challenge but can be applied to any size images. Potential challenges we would work through to do this would be developing labels, since the model classifies pixels in such a way that identifies structures, but the regions on interest in the DDSM data set are already extracted in one of the cleaner versions of the data set. We also read about ResNet, which was not specifically trained for biomedical applications, but is a widely known pre-trained model that allows much greater depth that previously possible and is widely applicable. Finally, we read about Conditional GANs, which can be used to generate captions for images that are more natural and diverse. This is a possible avenue for exploration, but it would likely require us coming up with "ground truth" captions, even if they don't especially correspond to images in the data set but rather just are more naturally structured, since a GAN discriminator must be fed both input from the generator and some "truth".

**Summary of model performance**: Jain and Levy [3] test multiple DCNN architectures on the DDSM dataset, and achieve 60.4% accuracy with a hand-built shallow CNN, 89.0% with AlexNet, and 92.9% with GoogleNet. Especially noteworthy is the fact that the GoogleNet's recall rate of 93.4% surpassed that of professional radiologists, who typically achieve recall rates that range from 74.5% to 92.3%. Shams et al. [7] perform simultaneous ROI-identification and classification using a model that combines CNNs and GANs, and achieve similar results of around 89% accuracy and 88.4% AOC on the DDSM. 

## Possible Methods of Interpretation/Explainability

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

## Data Background

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

## References

[1] Eric A. Scuccimarra, DDSM dataset, Version 10. Accessed at https://github.com/escuccim/mias-mammography

[2]  Scuccimarra, Eric A. “ConvNets for Detecting Abnormalities in DDSM Mammograms.” Medium, 21 May 2018, medium.com/@ericscuccimarra/convnets-for-classifying-ddsm-mammograms-1739e0fe8028.

[3] Arzav Jain, Daniel Levy. "DeepMammo: Breast Mass Classification using Deep Convolutional Neural Networks" Accessed at http://cs231n.stanford.edu/reports/2016/pdfs/306_Report.pdf)

[4] Heinrich Jiang, Been Kim, Melody Y. Guan, Maya Gupta. "To Trust Or Not To Trust A Classifier". Accessed at https://arxiv.org/abs/1805.11783.

[5] Aditya Khosla, Tinghui Zhou, Tomasz Malisiewicz, Alexei A. Efros, Antonio Torralba. "Undoing the Damage of Dataset Bias". European Conference on Computer Vision (ECCV), 2012. Accessed at http://undoingbias.csail.mit.edu.

[6] Pengcheng Xi, Chang Shu, Rafik Goubran. "Abnormality Detection in Mammography using Deep Convolutional Neural Networks." (March 5, 2018). Accessed at https://arxiv.org/pdf/1803.01906.pdf.

[7] Shayan Shams, Richard Platania, Jian Zhang, Joohyun Kim, Kisung Lee, Seung-Jong Park. "Deep Generative Breast Cancer Screening and Diagnosis." (Sept 26, 2018). Accessed at https://link.springer.com/chapter/10.1007/978-3-030-00934-2_95.





