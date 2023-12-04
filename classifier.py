import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

#get enoof data


data=np.array(pd.read_csv('train.csv'))
m, n = data.shape
np.random.shuffle(data) #shuffle up data before using

#define them, n,m are defined later on


data_dev =data[0:5000].T #transposing the data so label is on X axis, taking only 1000 features, nvm took 5k
Y_dev=data_dev[0]
X_dev=data_dev[1:n]/255. #the 256 values


data_train = data[5000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]/255. #yes 256 valoos again
_,m_train = X_train.shape

'''ok so 2 layers,input layer containing 784 units for 784 pixels respectively consider this a0
one hidden layer with 10 units with relu activation, consider this a1
and one output layer a2, with 10 nodes corresponding to 0-9 digits, with softmax



Forward prop : 
Z1 = W1 X + b1
A1 = relu (Z1)
Z2 = W2 A1 + b2
A2 = soft(Z2)               

backward prop:
dZ2 = A2 - Y
dW2 = 1/m dZ2 A1(transposed)
dB2 = 1/m summation(dZ2)
dZ1 = W2(transposed) dZ2 *g1(transposed) z1             NOTE *g1 is basically derivative of activation
dW1 = 1/m dZ1 A0 (transposed)                           NOTE all the 'd' variables stand for error, so basically label values - model predicted valeues
dB 1/m summation(dZ1)

variable updation:
W2 = W2 - alpha dW2
b2 = b2 - alpha db2
W1 = W1 - alpha dW1      alpha being some learning rate
b1 = b1 - alpha db1

ILL WRITE VARIABLE DEFINATIONS LATER
'''


#TRAINING 
def init_params():
    W1 = np.random.rand(10,784) - 0.5               # to maintain values with 0 as median and initialise rando data
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1,b1,W2,b2
#definiing activation functons to downscale the values to 0-1 floating poitn range
def ReLU(Z): # relu is easy just return x if x>0 else 0, the graph kinda looks similar to third approx. model of diode
    return np.maximum(Z,0)
def softmax(Z): #choosing softmax over tanh because i couldnt find eulers expansion for tanh, and i understand how softmax worked 
    A = np.exp(Z)/sum(np.exp(Z))
    return A



def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X)+b1 # dot product because they are arrays
    A1=ReLU(Z1) #input to hidden later activation is relu because ez to use and more than enough 
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2) # hidden to output layer activation is softmax to 
    return Z1,A1,Z2,A2
def ReLU_deriv(Z): # welp diff of 0 is useless, and lookin at the relu graph the slope(dy/dx) should be 1 as its linear 
    return Z>0  #this works because internally True/False are references as 1/0 which is what we need
def one_hot(Y): # one hot enconding to make the label a one dimensional array of length 9, with corresponding label value =1 rest =0, figuring out one hot encoding was the toughest :'')
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def param_update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): #update old variables wh
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_pred(A2): # output results for each iteration
    return np.argmax(A2,0)

def get_acc(predictions,Y):
    print(predictions,Y)  #prediction will be defined later
    return np.sum(predictions==Y)/Y.size
#logic for get_pred and get_acc was copy pastad and changeda bit
def gradient_descent(X,Y,alpha,iterations): #basically going back and forth bw forward and bacward propagation
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) #does intial value settin by tampering with random values set in init_params 
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) # bacward progataion using previous data 
        W1, b1, W2, b2 = param_update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) #finally setting the biasand weights right after every iteration of the descent
        if i % 10 == 0: #welp ill print every 10 iters because hmmm i dont want terminal spam 
            print("Iteration: ", i)
            predictions = get_pred(A2)
            print(get_acc(predictions, Y))
    return W1, b1, W2, b2


W1,b1,W2,b2=gradient_descent(X_train,Y_train,0.10,1000) # training the model for 1000 iterations welp it gave me 48 percent accuracy before i fixed issue with relu, now it gives me 80+
#also using actual data from csv file
# prediction on requested digit
def make_predictions(X, W1, b1, W2, b2): #function to make prediction from an iteration of forward propagation
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_pred(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    #converting pixel data into actual images
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

    time.sleep(3)

# plot dosent close auto because of some issue u can comment out below while loop and use test_prediction(int(input('enter some index(0-37000)')),W1,b1,W2,b2) multiple times to do multiple test cases

ch=0
while ch==0:
    test_prediction(int(input('enter some index(0-37000)')),W1,b1,W2,b2) #37000 because rest 5k i used on training
    plt.close('all')
    ch=1 if input('do you want to test more cases(y/n) ',W1,b1,W2,b2)=='n' else print('')
# test_prediction(8,W1,b1,W2,b2)
# test_prediction(5,W1,b1,W2,b2)
# test_prediction(1,W1,b1,W2,b2)
# test_prediction(3,W1,b1,W2,b2)
# test_prediction(2,W1,b1,W2,b2)
# test_prediction(9,W1,b1,W2,b2)

# test_prediction(0,W1,b1,W2,b2)

# test_prediction(7,W1,b1,W2,b2)


dev_predictions = make_predictions(X_dev, W1, b1, W2, b2) #getting accuracy value on dev set that was notused for training data
print(get_acc(dev_predictions, Y_dev))




