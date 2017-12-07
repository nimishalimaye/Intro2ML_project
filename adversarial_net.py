import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
import cPickle 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#from pprint import pprint
srng = RandomStreams()

TRAINING = True

# Convert into correct type for theano
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

# Weights are shared theano variables
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

# RMSProp to update weights
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

# Dropout regularization 
def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

# Neural network model, 3 fully connected layers
def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
	# Input layer: dropout + relu 
    X = dropout(X, p_drop_input)
    h = T.nnet.relu(T.dot(X, w_h))
	
	# Hidden layer: dropout + relu 
    h = dropout(h, p_drop_hidden)
    h2 = T.nnet.relu(T.dot(h, w_h2))
	
	# Output layer: dropout + softmax 
    h2 = dropout(h2, p_drop_hidden)
    py_x = T.nnet.softmax(T.dot(h2, w_o))
    return h, h2, py_x

print 'Loading MNIST data...'
trX, teX, trY, teY = mnist(onehot=True)

# Initialize theano variables for X, Y, and shared variables for weights
X = T.fmatrix()
Y = T.fmatrix()

if TRAINING:
    # For training of the net, we initialize weights to random values
    w_h = init_weights((784, 625))
    w_h2 = init_weights((625, 625))
    w_o = init_weights((625, 10))
    params = [w_h, w_h2, w_o]
else:
    # To run experiments, just read weights we learned before
    print 'Loading model...' 
    with open('LearnedParamsL1.model','rb') as fp:
        params = cPickle.load(fp)
    w_h, w_h2, w_o = params

# Dropout model for training
noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
# Use all-weights model for prediction
h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)

# To find confidence of test set use the following value of y_x
y_x1 = T.max(py_x, axis = 1)
# Define cost and update theano expressions

l1 = abs(w_h).sum() + abs(w_h2).sum() + abs(w_o).sum()
l2 = (w_h**2).sum() + (w_h2**2).sum() + (w_o**2).sum()

#=================== Parameters to chnge ===============================#
l1coef = 0.00001 #changed from 0.0001 to 0.01 NL1206 
l2coef = 0.00001 #changed from 0.0001 to 0.01 NL1206
#=======================================================================#


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y)) + l1coef * l1 + l2coef * l2
updates = RMSprop(cost, params, lr=0.001)

# Define train and predict theano functions
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
predict_conf = theano.function(inputs=[X], outputs=y_x1, allow_input_downcast=True)
if TRAINING:
    # Train in 50 epochs
    for i in range(50):
        # Select minibatch and train
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            cost = train(trX[start:end], trY[start:end])
        # Show test set accuracy. Its cost is not used for optimization,
        # it is just to show progress.
        print i, ':  ', np.mean(np.argmax(teY, axis=1) == predict(teX))
        print("In Training")
        # In each step save the learned weights
        with open('LearnedParamsL1.model','wb') as fp:
            cPickle.dump(params,fp)

#=========================== ADVERSARIAL COMPONENT OF THE CODE ============================#
            
#____________________________________________________________________________________________
#
# Now we have a trained model, either loaded or trained
# Time to create adversarial examples and test them 
            
# Theano function which calculates gradient of the cost function w.r.t. input image
cost_ad = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
get_grad = theano.function(inputs=[X, Y], outputs=T.grad(cost_ad, X), allow_input_downcast=True)
 


def plot_mnist_digit(image1, image2, name1, name2):
    global count_attack
    image1 = np.reshape(image1,[1,784])
    image2 = np.reshape(image2,[1,784])
    #print 'test image confidence' , np.mean(predict_conf(image1)), 'adversarial image confidence', np.mean(predict_conf(image2))
    if (predict(image1) != predict(image2)):
        count_attack = count_attack + 1


# Obtain output array with correct test outputs
Yts = []
for i in range(len(teY)):
    for j in range(10):
        if(teY[i,j] == 1):
            Yts.append(j)

#============================Parameter to change ========================================#
eps_values = [0.10, 0.25]  
#========================================================================================#

for EPS in eps_values:
    eps = EPS
    adX = []
    for i in range(len(teX)):
        gs = get_grad(teX[i:i+1], teY[i:i+1]).T[:,0]
        img_ad = teX[i] + eps * np.sign(gs) 
        adX.append(img_ad)

   # Find accuracy of the classifier on the test set and adversarial set
    pred_teY = predict(teX)
    print 'Test set Accuracy:					', np.mean(np.argmax(teY, axis=1) == predict(teX)) 
    print 'Adversarial set Accuracy, e=', eps, ':			', np.mean(np.argmax(teY, axis=1) == predict(adX))		#np.mean(predict(adX)!=Yts)
    print 'Test Set Confidence:					', np.mean(predict_conf(teX))
    print 'Adversarial set Confidence:				', np.mean(predict_conf(adX))
    count_attack = 0
    for i in range(len(teX)):     
       plot_mnist_digit(teX[i], adX[i], 'test_img{0}.jpg'.format(i),'less_ad_img{0}.jpg'.format(i))

    print 'Percent of successful adversarial attack:{0:3f}:							' .format(float(count_attack*100)/len(teX))
    print '================================================================================'




'''
try:
    with open('MNIST_test_adversarial_0007.bin', 'rb') as fp:
        adX = cPickle.load(fp)
except IOError: 
    print 'Adversarial examples not found, generating them now...' 
    # Eps is a parameter for strength of the added noise. 
    # Since MNIST dataset is binary, 0.25 error should be 'bellow' the resolution
    eps = 0.007
    adX = []
    for i in range(len(teX)):
        if (i % 1000 == 0):
            print i 
        gs = get_grad(teX[i:i+1], teY[i:i+1]).T[:,0]
        img_ad = teX[i] + eps * np.sign(gs) 
        adX.append(img_ad)
    with open('MNIST_test_adversarial_0007.bin', 'wb') as fp:
       cPickle.dump(adX, fp)
    
# Find accuracy of the classifier on the test set and adversarial set
print 'Test set:                ', np.mean(predict(teX))
print 'Adversarial set, e=0.007: ', np.mean(predict(adX))
 
try:
    with open('MNIST_test_adversarial_025.bin', 'rb') as fp:
        adX = cPickle.load(fp)
except IOError: 
    print 'The other file not found' 
print 'Adversarial set, e=0.25: ', np.mean(predict(adX))
 
# Let us display some images from both 

# Function for plotting a single MNIST image
count_attack = 0

def plot_mnist_digit(image1, image2, name1, name2):
    global count_attack
    image1 = np.reshape(image1,[1,784])
    #print(name1,predict(image1))
    image2 = np.reshape(image2,[1,784])
    #print(name2,predict(image2))
    if (predict(image1) != predict(image2)):
        #print(name1,predict(image1))
        #print(name2,predict(image2))
        count_attack = count_attack + 1

    image1 = np.reshape(image1, [28, 28])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image1, cmap=plt.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig(name1)
    image2 = np.reshape(image2, [28, 28])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image2, cmap=plt.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig(name2)

for i in range(len(teX)):     
    #plot_mnist_digit(teX[i], 'test{0}.jpg'.format(i))
    #plot_mnist_digit(adX[i], 'less_ad{0}.jpg'.format(i))
    plot_mnist_digit(teX[i], adX[i], 'test_img{0}.jpg'.format(i),'less_ad_img{0}.jpg'.format(i))

print 'Percent of successful adversarial attack:{0:3f}	'.format(float(count_attack*100)/len(teX))

'''
'''
# Generate random noise examples
eps = 0.25
rdX = []

for i in range(len(teX)):
    # Create random noise, zeros and ones, evenly distributed
    noise = np.random.binomial(1, 0.5, size=(784))
    # Turn into -1s and 1s
    noise = 2*noise - 1 
    img_rd = teX[i] + eps * noise 
    rdX.append(img_rd)

#for i in range(10):    
    #plot_mnist_digit(rdX[i], 'rd{0}.jpg'.format(i))
   
#print 'Random noise set, e=0.25:', np.mean(predict(rdX))
'''
    
    
    
    
    
    
    
