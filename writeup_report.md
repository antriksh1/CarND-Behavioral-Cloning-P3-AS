# **Behavioral Cloning** 

## Writeup Report

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center_2018_04_10_22_29_10_656.jpg "Center driving"
[image2]: ./center_2018_04_10_22_19_59_922.jpg "Right to Left correction"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py: containing the script to create and train the model 
* drive.py: for driving the car in autonomous mode
* model.h5: containing a trained convolution neural network 
* video.py: for recording the video
* writeup_report.md: summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```python drive.py model.h5```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with the following layers: (model.py - lines 68-71)

* Convolution Layer of depth: 6, with 5x5 filters
* Max Pooling (default 2x2)
* Convolution Layer of depth: 16, with 3x3 filters
* Max Pooling (default 2x2)

The model includes RELU layers to introduce nonlinearity - after each Convolution layer (code lines 68, 70).
The data is normalized in the model using a Keras lambda layer (code line 66).
The images are also cropped in the model using a Keras Cropping layer (code line 67). 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layers order to reduce overfitting (model.py lines 72). 
The dropout layer is right after the Convolution layers, so that convolution-data generated is dropped, before the flattenning and fully-connected (dense) layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 23, 53, 54, and 81). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also used the provided 'data', and the final model was trained on images from both mine, and provided 'data'.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to leverage the lessons learned in the previous project: Traffic Sign Classifier. 

My first step was to use a convolution neural network model similar to the LeNet used in the previous project. I thought this model might be appropriate because this was also an Convolution Neural Network based model for visual/image data, LeNet was my starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to introduce MaxPooling and Dropout. This dramatically helped with reducing the mse.

Then I re-ran the model, and it would constantly go into the water, right after the first turn, because it would steer less. This clearly meant that it was not 'sophisticated' enough.
So I increased the depth of the second Convolution layer from 6 to 16, and also increased the filter size of the first Convolution layer, from 3x3 to 5x5. Running this model, gave some good results, when the car was able to get to the bridge, but would run into the walls of the bridge.

At this point, I thought that LeNet is unable to handle the complexity of this task. So I programmed the NVIDIA-net, with 5 Convolution layers. This model also failed at the bridge, and would go near the edges of the road sometimes.

Then I thought, I need an even more complex model, and contacted the mentor, and got the suggestion to try out [this model here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). Even this model would be too near to the edges, and would fail either before or at the bridge.

Finally, someone from the discussion forums gave me the advise that I need to change the 'drive.py' file, to process 'bgr' instead of 'rgb' (drivey.py - line 67). Once I made that change, my models started working perfectly.

At this point, I wanted to reduce the model to be as simple as possible, and still pass Track-1. So I went back to my first LeNet model, and ran it, and my car completed Track-1 perfectly.

This proved to me that: small careless errors are far more damaging than a simple model, and that making things sophisticated (more layers, different models) is a brute-force approach.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65-77) consisted of a convolution neural network with the following layers and layer sizes:

* Intial Lambda to normalize the data around 0
* Cropping: 50 pixels from top, and 25 from the bottom
* Convolution Layer of depth: 6, with 5x5 filters
* Max Pooling (default 2x2)
* Convolution Layer of depth: 16, with 3x3 filters
* Max Pooling (default 2x2)
* Dropout of 0.4 (40% drop)
* Flattenning
* Fully Connected: Dense of 120 nodes
* Fully Connected: Dense of 80 nodes
* Fully Connected: Dense of 40 nodes
* Fully Connected: Dense of 1 nodes - to get the final steering angle output

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving, with occasional correction from edges built-into center-lane driving:

![Centre lane driving][image1]

Image showing: Centre lane driving

![Correction from right, while driving in center.][image2]

Image showing: Correction from right, while driving in center.

To augment the data set, I also flipped images and angles (model.py - lines: 46, 47) thinking that this would offset the fact that track-1 is mostly left-hand turns. Flipping images and negating the angles is an easy way to basically create another track with mostly right-handed truens, which would balance the data out. 

To augment the data set even further, I also used the left and right camera images , and I also corrected for the offset in angles by 0.2 (model.py - lines: 40-43). The goal here was to provide more data to the model, so it will make decisions from mode data.

After the collection process, I had around 12000 number of data points. I then preprocessed this data by normalizing it in the model's lambda layer and cropping it (model.py - lines: 66-67)

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by decreasing and finally reaching a very small loss of both training and validation. 
In addition, I noticed that running for 5 epochs would have been enough, but running for 8 ensured that loss would not rise or fluctuate again, which was the case with some of my initial models.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
