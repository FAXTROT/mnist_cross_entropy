# Welcome !

## What is this repo ?

Simple petproject Neural Network to classificate MNIST dataset ( hand-written digits from 0 to 9 included )

It has 4 layers : 1 input (28x28) , 2 hidden and 1 output (with 10 neurons as we predict 10 digits)
You can regulate them by variables l1 (first hidden), l2 (second hidden) and l3 (output)

Cross-entropy with backpropagation has been used

All activation functions are sigmoids

## How to use it ?

First install all required libs :
`$ pip install -r requirements.txt`

It would install mlxtend (dataset), matplotlib (visualization) and etc.

Then just run it with :
`$ python mnist.py`

It will show error, train accuracy and test accuracy

You can easily change quantity of epochs, alpha parameter and other parameters

