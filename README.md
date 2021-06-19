# Citrurs_Disease_Detection
Detection a Disease of Citrus tree is done using a pre-trained Convolution Neural Network with tensorflow and keras API

## Overview:-
* As the second-largest provider of carbohydrates in india it can withstand harsh conditions.
At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields.With the help of data science, it 
may be possible to identify common diseases so they can be treated.
* Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This 
suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since
African farmers may only have access to mobile-quality cameras with low-bandwidth.


In this project we're going to be using machine learning to help us identify different categories of Disease the plant have.
## Goal:-
  The main aim of This Project is to Classify The infected Images in to their Particular Disease category and provide the respected measure suggested by experties to that User.
## Data:-
>* This is dataset from Kaggle  
>* We have train set containing  more than 20k images, With train.csv file which contains id for image and labels.
To do this, we'll be using data from the [Kaggle ](https://www.kaggle.com/dtrilsbeek/citrus-leaves-prepared).   
The kaggle dataset is consists of a collection of 400+ trainig images and 121 validation images of 4 different citrus diseases.

In this data we've added the another two another classes i.e. Jamberry and mava and some images in healthy class. 

>* To know more about dataset i have added [dataset_info.md](https://github.com/AdiShirsath/Cassava-Leaf-Disease-Detection/blob/main/Dataset_info.md) file.



We're going to go through the following TensorFlow/Deep Learning workflow:
1. Get data ready (download from Kaggle, store, import).
2. Prepare the data (preprocessing, the 3 sets, X & y).
3. Choose and fit/train a model ([TensorFlow Hub](https://www.tensorflow.org/hub), `tf.keras.applications`, [TensorBoard](https://www.tensorflow.org/tensorboard), [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)).
4. Evaluating a model (making predictions, comparing them with the ground truth labels).
5. Improve the model through experimentation (start with subset of images, make sure it works, increase the number of images).
6. Save, sharing and reloading your model (once you're happy with the results).

For preprocessing our data, we're going to use `ImageDataGenerator`. The whole premise here is to get our data into Tensors (arrays of numbers which can be run on GPUs) and then allow a machine learning model to find patterns between them.

For our machine learning model, we're going to be using a pretrained deep learning model from TensorFlow Hub. 

The process of using a pretrained model and adapting it to your own problem is called **transfer learning**. We do this because rather than train our own model from scratch (could be timely and expensive), we leverage the patterns of another model which has been trained to classify images.

## Getting our workspace ready

## Preproccing
For preprocessing our data, we're going to use `ImageDataGenerator`. The whole premise here is to get our data into Tensors (arrays of numbers which can be run on GPUs) and then allow a machine learning model to find patterns between them.

* Converting image pixel's from 0-255 to 0-1 using tensorflow kept size of tensor (224,224,3) 
* Created tuples of images and respective labels with batch size of 32
* Preprocessed all data and saved it


## Training
We've Designed Lightweight Convolutional Neural Networks and selected Three Nets
* Mobile Net,Efficient Net,Nas Net and got a better accuracy on Mobilenet.
* ***Best Model***:-
>* Citrus-leaves-disease classifier whose architecture is same as MobileNetV3 
>* This was already trained of Cassava leaf's for 6 classes, Link to this model is here [Tensorflow](https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5)
>* We had five classes so trained this model by adding own fully connected layer.
>* Trained model is uploded [here](https://github.com/AdiShirsath/Cassava-Leaf-Disease-Detection/tree/main/Model)



## Fine Tuning
## Predictions
## Results
## Web App Interface







## ***Web-App Demo***:-



## Building Web-App :-

## Results:-
