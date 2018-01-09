# MNIST

I wanted to better understand how neural networks work and I'm a strong believer in learning by doing so I decided to build a neural network from scratch. I allowed myself to use NumPy and some varoius tools for data manipulation but no machine learning libraries. I've seen many people refer to hand-written digit recognition on the MNIST dataset as a great introduction to machine learning so I built and trained my Neural Network for that task, however this class can be used on any dataset (see modifications for other data). The purpose of this writeup is not to teach other people about neural networks or machine learning but instead for me to better understand what exactly I was doing and why. Also if I have to return to this later, this serves as great documentation. There are already far too many resources on machine learning out there and I'm sure nearly all of them are more accurate, thorough, and understandable than this.

#Approach

I used a simple feed-forward neural network because they are the most basic kind of neural network and also seem like the easiest to implement. Additionally, it seems like this model will give fairly good results on MNIST.

I used a sigmoid activation function and a log loss cost function and later MSE however I built the class so that other activations and costs could be used.

#Implementation
The ```shape``` attribute is a list of integers which describes the number of neurons at the ```i```th layer. Thus ```len(self.shape)```, which you'll see come up often, is the number of layers in the neural network.

It made the most sense to me to store the weights as matrices so that forward propagation takes the natural form of z=Wa+b. The ```weights``` attribute is a list of weights matrices where the ith element in ```weights``` represents the weights going from layer i-1 to layer i. Thus ```self.weights[0]``` is ```None``` which may seem strange but ultimately prevents having a bunch of ```i-1```s floating around. Similarly the ```bias``` attribute is a list of vectors where ```bias[i][j]``` represents the bias on the ```j```th neuron of the ```i```th layer.

Backprop is done using mostly NumPy vector operations to speed things up and make it more compact. The line right before the for loop is a little confusing so look at the markdown cell right after, which explains it.

#Modifications for Other Data

The class attribute ```inputChecks``` will check that various inputs to functions are the right size for using this neural network for MNIST. The purpose of having these checks is to help with debugging. However they can be turned off by setting this to false. This allows for using the neural network with other data or just debugging on smaller examples since 28^2 is a large dimension to deal with when debugging. 

#Results
I noticed that replacing loops with NumPy vector operations really sped things up. On any future projects I will definitely use that. There are a few places in my code currently where I can replace some loops with vector opertaoins but those functions are not used enough to warrant the changes I think.

At first I trained the neural network with log loss using stochastic gradient descent however around 66% accuracy, there was very little improvement. I switched to vanilla gradient descent however I noticed that gradient steps that decreased the total cost seemed to worsen the accuracy instead of improving it. Thus I stopped and switched to MSE and kept vanilla gradient descent and accuracy again began to improve.  


