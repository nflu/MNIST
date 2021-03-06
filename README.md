# MNIST

I wanted to better understand how neural networks work and I'm a strong believer in learning by doing so I decided to build a neural network from scratch. I allowed myself to use NumPy and some tools for data manipulation but no machine learning libraries. I've seen many people refer to hand-written digit recognition on the MNIST dataset as a great introduction to machine learning, so I built and trained my Neural Network for that task. The network can be used on any dataset (see modifications for other data). The purpose of this writeup is not to teach other people about neural networks or machine learning, but instead for me to better understand what exactly I was doing and why. I find that writing about what I did and why can reinforce my understnading. There are already far too many resources on machine learning out there and I'm sure nearly all of them are more accurate, thorough, and understandable than this.

# Approach

I used a simple feedforward neural network because they are the most basic kind of neural network (NN) and also seem like the easiest to implement. Additionally, it seems like this model will give fairly good results on MNIST.

I experimented with different activation functions, cost functions, and shapes. I ultimately found that pretty much any combination worked reasonably well (at least 80% accuracy) but more accuracy required fine-tuning. I ended up using ReLU for my activation function and softmax -> cross entropy for my loss function.

# Implementation
The ```shape``` attribute is a list of integers which describes the number of neurons at the ```i```th layer. Thus ```len(self.shape)```, which you'll see come up often, is the number of layers in the neural network.

It made the most sense to me to store the weights as matrices so that forward propagation takes the natural form of z=Wa+b. The ```weights``` attribute is a list of weights matrices where the ith element in ```weights``` represents the weights going from layer i-1 to layer i. Thus ```self.weights[0]``` is ```None``` which may seem strange but ultimately prevents having a bunch of ```i-1```s floating around. Similarly the ```bias``` attribute is a list of vectors where ```bias[i][j]``` represents the bias on the ```j```th neuron of the ```i```th layer.

Backprop is done using mostly NumPy vector operations to speed things up and make it more compact. The line right before the for loop (and the similar line inside the for loop) is a little confusing so look at the markdown cell right after, which explains it.

# Modifications for Other Data

The class attribute ```inputChecks``` will check that various inputs to functions are the right size for using this neural network for MNIST. The purpose of having these checks is to help with debugging since Python's dynamic-typing can be a little confusing when trying to diagnosis problems regarding incorrectly formatted data. However they can be turned off by setting this to false. This allows for using the neural network with other data or just debugging on smaller examples since 28^2 is a large dimension to deal with when debugging. 

# Results
I noticed that replacing loops with NumPy vector operations really sped things up. On any future projects I will definitely use that. There are a few places in my code currently where I can replace some loops with vector operations but those functions are not used enough to warrant the changes.

I was surprised by how much fine-tuning I needed to reach higher levels of accuracy but I think that the blogs post, articles, and forum questions that I read on the topic were very benificial. It seems like hyper-parameter tuning is what the majority of a developer's time goes to when working with these types of models.

I was similarly surprised by how well even the simplest model worked. Also even using MSE as a cost function worked very well (by my standards) despite that this is a classification problem. 

Stochastic gradient descent also was a great tool. I was surprised how low I could set the batchsize and still see consistent improvement in the model over multiple iterations. Using SGD really speeds up training and I was impressed with its power. I should definitely read about the theoretical justifications for its effectiveness.

There were also times when I was concerned that the model was not working but I just needed to let it train for more steps. I was running my model for iterations on the order of hundreds whereas I saw others used at least ten times that. I think I didn't quite appreciate how long training deep learning models takes before this project.

Also it is super important that you have some way to save the configuration of your model after training. I accidentally lost hours worth of a trained model.

Good luck future self (or whoever is reading this) with training your models :) 

# How to View Results
I saved the training data and best weights and biases I found so far in a dictionary called ```training_data``` and using pickle, saved it in the file ```training_data.pickle``` . The current code on notebook demonstrates how to load the weights and biases and examine the training data.
  
# Thinking in Terms of Optimization
In all the things I read about machine learning I never once saw this approach and since I had taken convex optimization just before doing this, I found this view very elucidating.

We would like to minimize the number of data points our model incorrectly classifies on all the data it will see when in production. We don't have access to all the data our model will ever see in production, so instead we use a training data set that we hope is representative of what our model will be used on.

The problem now is that our objective function is non-differentiable since it is the number of data points the model misclassifies. So we use a differentiable cost function that approximates the number of missclassified points. That's why people try different cost functions because the cost function is itself a hyper-parameter that must be tuned to get the best accuracy on the dataset.

Then we just use gradient-based optimization techniques to minimize that objective and our decision variables are the weights and biases of the model which determine its shape. Backprop is simply a way to calculate the gradient of the cost function with respect to the weights and biases.


