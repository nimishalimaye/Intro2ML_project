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
    #with open('LearnedParamsL1.model','rb') as fp:
        #params = cPickle.load(fp)
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

#==========================Training for blue team==========================#

#=================== Parameters to change ===============================#
l1coef = [ 0.0, 0.0001, 0.00001 ] 
l2coef = [ 0.0, 0.0001, 0.00001 ] 
#=======================================================================#

for l in l1coef:
    for j in l2coef:
        print("l1coef = %f, l2coef = %f" %(l,j))
        cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y)) + l * l1 + j * l2
        updates = RMSprop(cost, params, lr=0.001)

        # Define train and predict theano functions
        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
        predict_conf = theano.function(inputs=[X], outputs=y_x1, allow_input_downcast=True)
        
        if TRAINING:
            print('Training MNIST data...')
            # Train in 5 epochs
            for k in range(5):
                # Select minibatch and train
                for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
                    cost = train(trX[start:end], trY[start:end])
                # Show test set accuracy. Its cost is not used for optimization,
                # it is just to show progress.
                print(k, ':  ', np.mean(np.argmax(teY, axis=1) == predict(teX)))
                # In each step save the learned weights
                #with open('LearnedParamsL1.model','wb') as fp:
                    #cPickle.dump(params,fp)
            print("Accuracy is:    ",np.mean(np.argmax(teY, axis=1) == predict(teX)))
            print("Confidence is:    ", np.mean(predict_conf(teX)))
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
	    #if (predict(image1) != predict(image2)):
		#count_attack = count_attack + 1

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
	    #print 'Test set Accuracy:					', np.mean(np.argmax(teY, axis=1) == predict(teX)) 
	    print 'Adversarial set Accuracy, e=', eps, ':			', np.mean(np.argmax(teY, axis=1) == predict(adX))		#np.mean(predict(adX)!=Yts)
	    #print 'Test Set Confidence:					', np.mean(predict_conf(teX))
	    print 'Adversarial set Confidence:				', np.mean(predict_conf(adX))
	    #count_attack = 0
	    #for i in range(len(teX)):     
	       #plot_mnist_digit(teX[i], adX[i], 'test_img{0}.jpg'.format(i),'less_ad_img{0}.jpg'.format(i))

	    #print 'Percent of successful adversarial attack:{0:3f}:							' .format(float(count_attack*100)/len(teX))
	    print '================================================================================'

'''
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
        #print("In Training")
        # In each step save the learned weights
        with open('LearnedParamsL1.model','wb') as fp:
            cPickle.dump(params,fp)
print("Accuracy is:    ",np.mean(np.argmax(teY, axis=1) == predict(teX)))
print("Confidence is:    ", np.mean(predict_conf(teX)))

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        #Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

'''

'''
noises = ["gauss","s&p",'poisson','speckle']
for noise in noises:
    print("For noise: ",noise)
    adX = []
    for i in range(len(teX)):
        image = np.resize(teX[i],(28,28,1))
        img_ad = teX[i] + noisy(noise,image) 
        adX.append(img_ad)
    print("Accuracy after adding noise is:    ",np.mean(np.argmax(teY, axis=1) == predict(adX)))
    print("Confidence after adding noise is:    ", np.mean(predict_conf(adX)))
'''
