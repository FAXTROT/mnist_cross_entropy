# prepare libs and data

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import mnist_data

X, Y = mnist_data()

X = X/255

#remain_idx = [idx for idx in range(0, 5000) if idx%10 != 0] # cut data set
#X, Y = X[remain_idx], Y[remain_idx]

print(X.shape)
print(Y.shape)
print("OK")

# divide into train and test sets properly

train_idx = [idx for idx in range(0, 5000) if idx%5 != 0]

x_train, y_train = X[train_idx], Y[train_idx]

test_idx = [idx for idx in range(0, 5000) if idx%5 == 0]

x_test, y_test = X[test_idx], Y[test_idx]

print(np.bincount(y_test))
print(np.bincount(y_train), '\n')

print(np.unique(y_test))
print(np.unique(y_train), '\n')

print("Train set : ")
print(x_train.shape)
print(y_train.shape, '\n')

print("Test set : ")
print(x_test.shape)
print(y_test.shape)

# plot single example from any set
def plot(x, y, idx):
    img = x[idx].reshape(28, 28)
    plt.imshow(img, cmap="Greys", interpolation='nearest')
    plt.title("Label is " + str(y[idx]))
    plt.show()
    
plot(x_test, y_test, 499)

# hot spots
y_temp = np.zeros([y_train.shape[0], 10])
print(y_temp.shape)
for i, val in enumerate(y_train):
    y_temp[i, val] = 1

y_train = y_temp  

y_temp = np.zeros([y_test.shape[0], 10])
print(y_temp.shape)
for i, val in enumerate(y_test):
    y_temp[i, val] = 1

y_test = y_temp  


# activation functions and their deriv

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))
    
def relu(z):
    return z*(z > 0)

# init weights and etc.

alpha = 0.6

m, n = x_train.shape

l1 = 30
l2 = 30
l3 = 10

w1 = 2*np.random.random([n, l1]) - 1
w2 = 2*np.random.random([l1, l2]) - 1
w3 = 2*np.random.random([l2, l3]) - 1

b1 = np.ones([1,l1])
b2 = np.ones([1,l2])
b3 = np.ones([1,l3])

# for plot
error = []
train_acc = []
iterations = []
test_acc = []

print('\n')
# body
for i in range(0, 100):      # quantity of epochs = range(0, NUMBER)
    iterations.append(i)
    
    # --------------------
    # test acc
    z1 = np.dot(x_test, w1) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    
    z3 = np.dot(a2, w3) + b3
    a3 = sigmoid(z3)
    
    counter = 0
    pred = np.argmax(a3, axis=1)
    for c, val in enumerate(y_test):
        if val[pred[c]] == 1:
            counter += 1
    test_acc_value = counter*100/y_test.shape[0]
    test_acc.append(test_acc_value)
    # --------------------
    
    # feed forward
    z1 = np.dot(x_train, w1) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    
    z3 = np.dot(a2, w3) + b3
    a3 = sigmoid(z3)
    
    J = np.sum((a3-y_train)**2)/(2*m)
    #J = -1*np.sum( y_train*np.log(a3+0.00001) + (1-y_train)*np.log(1-a3+0.00001) )/m
    error.append(J)
        
    
    # backprop
    d3 = a3-y_train
    d2 = np.dot(d3, w3.T)
    d1 = np.dot(d2, w2.T)
    
    w3 = w3 - alpha*np.dot(a2.T, d3)/m
    w2 = w2 - alpha*np.dot(a1.T, d2)/m
    w1 = w1 - alpha*np.dot(x_train.T, d1)/m
    
    b1 = b1 - alpha*np.sum(d1, axis=0)/m
    b2 = b2 - alpha*np.sum(d2, axis=0)/m
    b3 = b3 - alpha*np.sum(d3, axis=0)/m
    
    # count
    counter = 0
    pred = np.argmax(a3, axis=1)
    for c, val in enumerate(y_train):
        if val[pred[c]] == 1:
            counter += 1
    train_acc.append(counter*100/m)
            
    if i%10 == 0:
        print(f"Iteration No_{i}")
        print(f"Error : {J}")
        #print(w3)
        #print(b3)
        #print(a3)
        print(f"Train accuracy : {counter*100/m}")
        print(f"Test accuracy : {test_acc_value}")
        print('\n')

# plot
plt.plot(iterations, error, 'r-')
plt.plot(iterations, train_acc, 'y-')
plt.plot(iterations, test_acc, 'g-')
plt.legend(labels=("J error", "train accuracy", "test accuracy"), loc="upper left")
plt.xlabel("iterations")
plt.title("Graphic")
plt.show()

plt.plot(iterations, error, 'r-')
plt.xlabel("Iterations")
plt.ylabel("Error value")
plt.title("Error (J)")
plt.show()


