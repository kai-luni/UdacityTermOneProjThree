**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

# Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

## Required Files

### Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

## Quality of Code

### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


## Model Architecture and Training Strategy

### An appropriate model architecture has been employed
My model consists of a convolution neural network with varying filter sizes and depths between 24 and 64 (model.ipynb third cell) 

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer. 

### Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.  

The model was trained in a large dataset with around 100000 pictures to further insure that it would not overfit. The brightness was always slightly changed on each picture to get more variation.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving the track in the opposite direction.

## Architecture and Training Documentation

### Solution Design Approach
The overall strategy for deriving a model architecture was to roughly follow the Nvidia model and follow a try and error approach. I used this approach because it was recommended in the udacity class.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a satisfying accuracy of about 98%, but on the road it failed on the first curve.

The first approach to imrpove the behaviour was to create more training data and train it for more epochs. Even the accuracy would get higher than 99%, the driving behavior was not satisfying.

The step were it started working was when I added Relu activation functions to the dense layers and changed the strides of the convolutional layers in a way that the last convolutional layer has a output shape of (none, 6 ,5 ,64) instead of (none, 2, 9, 64). It is not clear to me which of the two brought the success.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Final Model Architecture

<table>
 <tr>
  <td>Input</td>
  <td>shape: 160x320x3 </td>
 </tr>
 <tr>
  <td>Lambda</td>
  <td>Shape: Normalize range -0.5<=>0.5 </td>
 </tr>
 <tr>
  <td>Cropping</td>
  <td>output: 90x320x3 </td>
 </tr>
 <tr>
  <td >Convolution 5x5</td>
  <td>Stride:2x3, padding:'VALID', output:43x106x24 </td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Dropout</td>
  <td>keep:60%</td>
 </tr>
 <tr>
  <td> Convolution 5x5</td>
  <td>Stride:1x2, padding:'VALID', output:39x51x36 </td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Dropout</td>
  <td>keep:70%</td>
 </tr>
 <tr>
  <td>Convolution 4x4</td>
  <td>Stride:2x3, padding:'VALID', output:18x16x48 </td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Dropout</td>
  <td>keep:70%</td>
 </tr>
 <tr>
  <td>Convolution 3x3</td>
  <td>Stride:2x2, padding:'VALID', output:8x7x48 </td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Convolution 3x3</td>
  <td>Stride:1x1, padding:'VALID', output:6x5x64 </td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Flatten</td>
  <td>output:1920</td>
 </tr>
 <tr>
  <td>Fully Connected</td>
  <td>output:1000</td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Fully Connected</td>
  <td>output:100</td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Fully Connected</td>
  <td>output:50</td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Fully Connected</td>
  <td>output:10</td>
 </tr>
 <tr>
  <td>RELU</td>
  <td></td>
 </tr>
 <tr>
  <td>Fully Connected</td>
  <td>output:1</td>
 </tr>
</table>
####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
