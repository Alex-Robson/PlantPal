# PlantPal

![alt text](https://github.com/Alex-Robson/PlantPal/blob/master/data/streamlit/Logo.png?raw=true)

[PlantPal](PlantPal.org) is a web app that rapidly identifies houseplants from photos and provides basic care information and information on any potential toxicity to the users pets. Potentially saving them from years of discomfort or serious illness.

# Give it a try
The web app and algorithm are hosted at [PlantPal.org](PlantPal.org)

It should work something similar to this:   
<img src="https://github.com/Alex-Robson/PlantPal/blob/master/figures/demo.gif?raw=true" width="1200px">

# Bug tracker
Have a bug? Please create an issue here on GitHub at https://github.com/Alex-Robson/PlantPal/issues.

# Requirements
Requirements can be found in requirements .txt      
They can be installed to your environment with pip:     
     `pip install -r requirements.txt`
    
# Scraping

Using notebooks 01-04 you can scrape houseplant images from [Google Images](https://www.google.com/search?q=fiddle+leaf+fig&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjO3J6Cj9TsAhXRjp4KHdVdAZ4Q_AUoAnoECDcQBA&biw=1237&bih=786), [Shutterstock](https://www.shutterstock.com/search/fiddle+leaf+fig?image_type=photo), and [Pl@ntNet](https://identify.plantnet.org/reunion/species/Ficus%20lyrata%20Warb./data#) (a database of user submitted images), before automatically removing duplicated

# EDA and Training

Notebook 05_EDA.ipynb primarily explores and visualizes the data imbalance in the current dataset
![alt text](https://github.com/Alex-Robson/PlantPal/blob/master/data/figures/number_imgs_per_class.png?raw=true)

06_train_network.ipynb shows examples of the data, their preprocessing/augmentation, and the resulting training.       
         
Training was done via transfer learning with models such as [Inception Resnet v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2), [Inception v3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3), [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50), and [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16) starting with the imagenet weights. For the first 20 epochs these were frozen and only a new training block was free to train. After these initial 20 epochs the whole model was unfrozen and the learning rate reduced before another 80 epochs of training.

# Model evaluation

Finally 07_model_evaluation.ipynb compares how the different models metrics (validation accuracy and top 5 accuracy) vary throughout training, as well as exploring the confusion matrix and other useful information for model evaluation.     
![alt text](https://github.com/Alex-Robson/PlantPal/blob/master/figures/comparison.png?raw=true)
Ultimately, most models considered, including the final model built using [Inception Resnet v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2) produced ~95% validation accuracy. More imporantlty the model shows ~99.5% top 5 accuracy, meaning only 1/200 images of plant which are present in the database are not offered to the user to select from the dropdown - in which case they can select "None of the above" and the app will provide tips on providing a more readily identified photo.

# Web app
The web app is built using [streamlit](https://www.streamlit.io/) and is hosted on AWS at [PlantPal.org](PlantPal.org).   
The final model implemented is [Inception Resnet v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2)
      
