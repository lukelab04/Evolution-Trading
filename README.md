# Evolution-Trading
A small example to show how stock trading can be done with evolutionary neural networks.

A more in-depth explanation of this code can be found here: 

# Usage

You should be able to get a pretty good idea of how to use this by watching the video / reading the code, but I am going to add a few things here that you should keep in mind.

First: When giving training data to the network, you should be supplying it with a 1 dimensional array, not a two dimensional array. In the source code, the training data is supplied by looping though the stockData array (which is 2D) and inputting each sub array to the network. The size of these arrays should match the input layer of the network. For example, the network in the source code has 5 inputs, and each training set has 5 elements.

Second: The output of the network will be a one dimensional array the same size as the output neurons. This array corresponds directly with the values of the last layer of neurons.

Third: The 'structure' parameter of the model constructor should be a one dimensional array holding the number of neurons in each layer. For example, to create a network with 3 inputs, 1 hidden layer of 6 neurons, and 2 outputs, the structure would be [3, 6, 2].

Fourth: THIS IS FOR DEMONSTRATION. I went through this in the video as well, but the example in this code is absolutely not something you should be using to do any real trading with whatsoever. This is meant more as a guide to help you understand how evolutionary neural networks might be used in a trading environment.

That should be everything you need to get started. Feel free to leave a comment on the youtube video if you need help and I will respond when I can.
