# SomeNeuralNet
Numpy implementations of key concepts and algorithms in deep learning. 
## Motivation
I chose to write this small Python library for the purpose of learning deep learning starting from the acquisition of a low-level knowledge of the algorithms in this field.
This small personal project underwent between December '23 and February '24. The implementations of the various layers occurred sparsely during this period.  
The implementations should be correct and work properly. There is a lot of naive coding and some things are not good coding examples (e.g. the way model parameters are saved/loaded).  
I do not recommend to use this code for deep learning projects.   
The hope is that this repository might somehow help other people who desire to directly implement deep-learning algorithms.   
My suggestion is to do it, since it allows you to develop a good comprehension of the topics, enough for being able to think at further developments or innovations. Moreover, it is a relatively easy task, especially if you already have (even basic) knowledge of linear algebra.
## What you will find
Here, you will find some implementations of key stuff in deep learning. Everything is implemented in numpy. If cupy  is installed, it can be used instead of numpy. Gradients are computed by hand. The structure of layer and network classes is described at line 167 of "neural_network_library.py".  

What is implemented:
* Loss and activation functions: squared error loss, multi-class cross entropy loss, sigmoid activation, relu activation, tanh activation
* Optimizers: Adam.
* Layers:
  - fully connected layer
  - batch normalization
  - layer normalization
  - dropout
  - softmax layer
  - embedding layer
  - positional encoding
  - multi-head attention mechanism
  - residual block
* Some helper functions.
