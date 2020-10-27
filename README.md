# PlantPal

![alt text](https://github.com/Alex-Robson/PlantPal/blob/master/data/streamlit/Logo.png?raw=true)

# Give it a try
The web app and algorithm are hosted at [PlantPal.org](PlantPal.org)

It should work something similar to this:   
+<img src="https://github.com/Alex-Robson/PlantPal/blob/master/figures/demo.gif?raw=true" width="1200px">

# Bug tracker
Have a bug? Please create an issue here on GitHub at https://github.com/Alex-Robson/PlantPal/issues.

# Requirements
Requirements can be found in requirements .txt      
They can be installed to your environment with pip:     
     `pip install -r requirements.txt`
    
# Scraping

Using notebooks 01-04 you can scrape houseplant images from Google Images, Shutterstock, and Pl@ntNet (a database of user submitted images), before automatically removing duplicated

# EDA and Training

Notebook 05_EDA.ipynb primarily explores and visualizes the data imbalance in the current dataset, while 06_train_network.ipynb shows examples of the data, their preprocessing/augmentation, and the resulting training.       
         
Training was done via transfer learning with models such as Inception Resnet v2, Inception v3, ResNet50, and VGG16

# Model evaluation

Finally 07_model_evaluation.ipynb compares how the different models metrics (validation accuracy and top 5 accuracy) vary throughout training, as well as exploring the confusion matrix and other useful information for model evaluation.      
       
A plant app, that generates a personalised set of care instructions (and google calendar link) based on images of your house plants.
