# Neural Network

This is my attempt at creating a library from scratch that builds feedforward neural networks to help understand more deeply how they work. I have developed this concurrently with my Machine Learning Lab application where I will, fingers crossed, be able to deploy this to a full stack app. It is my hope that anyone who comes across this can look at the code project along with this readme and feel like they at least have some idea where to start when it come to trying replicate something like this. 

## My Goal
I learn best through having a goal so I decided to attempt to build a network that would be able to classify the the sklearn iris dataset with above 95% accuracy. 

## Resources 
I could not have attempted this without the wonderful resources made available to the community. Some of the most helpful resources for me include [Professor Daniel Shiffman's Youtube Series](https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh) where he explains the mathmatics behind and builds a 3 layer neural network in JavaScript. Another fantastic resource was the [Video series by 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) where the theory of Neural Networks is discussed in depth. Another fantastic reference is [Neural Networks And Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen. If you, like I did,  would like to bone up on some of the  math concepts required like Linear Algebra (matrix layouts, scalar multiplication, element-wise addition, dot-product) then check out Khan Academy's [Linear Algebra Playlist](https://www.youtube.com/watch?v=xyAuNHPsq-g&list=PLFD0EB975BA0CC1E0). In addition there is some Calculus concepts like gradients and the chain rule that you should feel comfortable with. Again, Khan Acadmey comes in clutch to assist with their [Calculus Playlists](https://www.youtube.com/watch?v=TrcCbdWwCBc&list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7)

## How Neural Networks Work
See one, do one, teach one is the motto so I'm gonna try to give you a overview of how these things work. I'm first going to cover what the architecture a network is. I'm then going to cover how the network can "learn". Finally, I'm going to go over one way to train a network. There is much more commentary within the comments in the code, so check that out as well.

### Architecture 
So what do networks even look like? 
![image](https://user-images.githubusercontent.com/46241098/75271864-2127a000-57ba-11ea-9e36-77251ea3872b.png)

Notice the little circles, which are called neurons and lines, which I'm going to refer to as weights. Look at how all the neurons in each layer have connections, that are associated with the weights, to every node in the next layer. These weights along with biases (discussed later) are tuned to help the network learn.

A network has 3 kinds of layer, an input layer (the first layer), an output layer (the last layer), and every layer inbetween is called a hidden layer. If we imagine a 3 layer network. Meaning there is an input layer which i'm going to call I, one hidden layer of  neurons which I will call layer H and an output layer, which is denoted as layer O. The data will flow through like this The all inputs in layer I will be feed into every neuron in layer H. Every output produced by by the neurons in layer H, will be feed into every neuron in layer I. The outputs of the nuerons in layer I represent the answer of the network. 

This nature of forward flow of the inputs through the network is why we call this a feed forward network. So what are the neurons? Neurons are a function, meaning they take input, preform some operation 
on the input, and return an output. Yeah, sick but what function are the neurons preforming, what are they doing it on, and what's the output look like? 

Neurons preform a weighted sum using all inputs and adds a bias, then applies some function called the activation function to that sum, and then returns a single output. This output becomes an input 
for the next layer in the network. Nuerons in every layer execpt the first will do this operation. I'll break down the bias, and the activation function since we've already covered the weights.

The bias described above is simply a number that will be added to every sum. Every neuron except the input neurons have a bias associated with them. 

The activation function is a function that will be able to map the weighted sum plust that bias to a value in the range [-1 , 1].

### Learning

**Assume all of these are matrices**

So when we say a network is learning what does that mean? Really this is the process of comparing a known answer to the answer that the network output. Then this error is used to calculate the changes to be made to the weights and biases. 

Calculating the error is easy for the output layer we just subtract the predicted value from the actual value. But then the question becomes well how do I get the error for the rest of the layers. Well the error for all neurons in a layer is the dot product of the transposed weights matrix for the layer and the error of the neurons for the layer prior. 

Next we will calculate the amount that we should adjust the weights and biases for a single layer. Before we do this we must introduce the concept of the learning rate and talk about derivaties for a second. A learning rate is a scalar multiplier that well temper the size of adjustment that we will make to weights and biases. Next for this step we will need to feed all outputs from a layer throught the derivative of the activation function for the layer, then multiply the result by the learning rate and the error for the layer. The deltas for the weights in a layer is the dot product of the transposed wieghts for a layer and the gradients calculated earlier. To adjust the weights and biases add the weights to the deltas and the biases to the gradients. 

This is a gross over simplifiication and is explained much more in the references above which you should check out. 


### Training
Stochastic Gradient Descent! Man what a mouth full. We will be training our network using this method. To do this we will be, over a set number of iterations, randomly picking a element from a subset of the inputs and outputs. Adjusting the weights and biases like we talked about previously, and evaluating our network using the remainder of the inputs and outputs to test how many our network correctly identifies. This is a much more indepth topic, but for the purposes of making the network this is all the intuiton needed. 



## The Technologies Used

While it was my intention to build this without using a complete framework I still usedseveral popular python libraries to assist with this goal. I mean come on, I'm not actuallygonna do this all from scratch. This whole project was worked
out in tested in a jupyter Notebook. I used sklearn's datasets and joblib library to get my dataset and subsequently to save my network after I had finished. I used both numpy to handle the actualy matrix storage and mathmatical manipulations
and manipulations and pandas to assist with storing data. 

## Dependencies 
To get the class working you will need sklearn and numpy. I do recommend installing pandas as well if you intend to implement the Network class. Also I would recommend installing Jupyter Notebook to play around with the class in a live environment.
```sh
$ pip install notebook
$ pip install numpy
$ pip install scikit-learn
$ pip install pandas
$ pip install joblib
```