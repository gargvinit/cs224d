
# coding: utf-8

# # Semantic Word Vectors and Sentiment Analysis
# CS 224D Assignment 1  
# Spring 2015
# 
# *Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://cs224d.stanford.edu/assignment1) on the course website.*
# 
# In this assignment, we will walk you through the process of implementing 
# 
# - A softmax function
# - A simple neural network
# - Back propagation
# - Word2vec models
# 
# and training your own word vectors with stochastic gradient descent (SGD) for a sentiment analysis task. Please make sure to finish the corresponding problems in the problem set PDF when instructed by the worksheet.
# 
# The purpose of this assignment is to familiarize you with basic knowledge about neural networks and machine learning, including optimization and cross-validation, and help you gain proficiency in writing efficient, vectorized code.
# 
# ** Please don't add or remove any code cells, as it might break our automatic grading system and affect your grade. **

# **Honor Code:** I hereby agree to abide the Stanford Honor Code and that of the Computer Science Department, promise that the submitted assignment is my own work, and understand that my code is subject to plagiarism test.
# 
# **Signature**:Vint Garg

# In[4]:

# Run some setup code for this notebook. Don't modify anything in this cell.

import random
import numpy as np
from cs224d.data_utils import *
import matplotlib.pyplot as plt
import math
from sklearn.tree._tree import Gini
# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
# get_ipython().magic(u'matplotlib inline')
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')


# ## 1. Softmax
# *Please answer the first first complementary problem before starting this part.*
# 
# Given an input matrix of *N* rows and *d* columns, compute the softmax prediction for each row. That is, when the input is
# 
#     [[1,2],
#     [3,4]]
#     
# the output of your functions should be
# 
#     [[0.2689, 0.7311],
#     [0.2689, 0.7311]]

# In[5]:

def softmaxactual(x): 
    maxX = x.max()
    exp = np.exp(x - maxX)
    return exp / sum(exp)
def softmax(x):
    """ Softmax function """
    ###################################################################
    # Compute the softmax function for the input here.                #
    # It is crucial that this function is optimized for speed because #
    # it will be used frequently in later code.                       #
    # You might find numpy functions np.exp, np.sum, np.reshape,      #
    # np.max, and numpy broadcasting useful for this task. (numpy     #
    # broadcasting documentation:                                     #
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)  #
    # You should also make sure that your code works for one          #
    # dimensional inputs (treat the vector as a row), you might find  #
    # it helpful for your later problems.                             #
    ###################################################################
    
    return map(softmaxactual, x)


# In[6]:

# Verify your softmax implementation

print "=== For autograder ==="
print softmax(np.array([[1001, 1002], [3, 4]]))
print softmax(np.array([[-1001, -1002]]))


# ## 2. Neural network basics
# 
# *Please answer the second complementary question before starting this part.*
# 
# In this part, you're going to implement
# 
# * A sigmoid activation function and its gradient
# * A forward propagation for a simple neural network with cross-entropy cost
# * A backward propagation algorithm to compute gradients for the parameters
# * Gradient / derivative check

# In[17]:

def sigmoid(x):
    import math
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################
    
    return 1. / (1. + np.exp(-x))
    
    

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    ###################################################################
    # Compute the gradient for the sigmoid function here. Note that   #
    # for this implementation, the input f should be the sigmoid      #
    # function value of your original input x.                        #
    ###################################################################
    
    return sigmoid(f) * (1. - sigmoid(f))

# In[18]:

# Check your sigmoid implementation
x = np.array([[1, 2], [-1, -2]])
f = sigmoid(x)
g = sigmoid_grad(f)
print "=== For autograder ==="
print f
print g


# Now, use the functions you just implemented, fill in the following functions to implement a neural network with one sigmoid hidden layer. You might find the handout and your answers to the second complementary problem helpful for this part.

# In[24]:

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        random.setstate(rndstate)  
        x[ix] += h
        fxph, grad1 = f(x)
        random.setstate(rndstate)  
        x[ix] -= 2*h
        fxmh, grad2 = f(x)
        numgrad = (fxph-fxmh) / (2.0 * h)
        x[ix] += h
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
    
        it.iternext()  # Step to next dimension

    print "Gradient check passed!"


# In[25]:

# Sanity check for the gradient checker
quad = lambda x: (np.sum(x ** 2), x * 2)

print "=== For autograder ==="
gradcheck_naive(quad, np.array(123.456))  # scalar test
gradcheck_naive(quad, np.random.randn(3,))  # 1-D test
gradcheck_naive(quad, np.random.randn(4, 5))  # 2-D test


# In[ ]:

# Set up fake data and parameters for the neural network
N = 20
dimensions = [10, 5, 11]
data = np.random.randn( dimensions[0],N)  # each row will be a datum
labels = np.zeros((dimensions[2],N))
for i in xrange(N):
    labels[ random.randint(0, dimensions[2] - 1),i] = 1

params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2],)


# In[ ]:

def forward_backward_prop(y0, labels, params):
    """ Forward and backward propagation for a two-layer sigmoidal network """
    ###################################################################
    # Compute the forward propagation and for the cross entropy cost, #
    # and backward propagation for the gradients for all parameters.  #
    ###################################################################
    
    # ## Unpack network parameters (do not modify)
    t = 0
    W1 = np.reshape(params[t:t + dimensions[0] * dimensions[1]], ( dimensions[1],dimensions[0]))
    t += dimensions[0] * dimensions[1]
    b1 = np.reshape(params[t:t + dimensions[1]], (dimensions[1],1))
    t += dimensions[1]
    W2 = np.reshape(params[t:t + dimensions[1] * dimensions[2]], (dimensions[2],dimensions[1]))
    t += dimensions[1] * dimensions[2]
    b2 = np.reshape(params[t:t + dimensions[2]], ( dimensions[2],1))
    # ## YOUR CODE HERE: forward propagation
    #http://www.willamette.edu/~gorr/classes/cs449/backprop.html
 
    N =np.shape(y0)[1]
    ones = np.ones((1,N))
    a1=np.dot(W1,y0 ) + np.dot(b1, ones)
    y1 = sigmoid(a1)
    a2=np.dot(W2,y1 ) + np.dot(b2, ones)
    y2 = np.transpose(softmax(np.transpose(a2)))
    d2 = y2-labels
    d1=np.multiply(np.dot(np.transpose(W2), d2),sigmoid_grad(a1))

    gradW2 = np.dot(d2, np.transpose(y1))
    gradb2 = np.dot(d2, np.transpose(ones))
    gradW1 = np.dot(d1, np.transpose(y0))
    gradb1 = np.dot(d1, np.transpose(ones))
    cost =sum(sum(np.multiply(labels,-np.log(y2))))
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    return cost, grad


# In[ ]:

# Perform gradcheck on your neural network
print "=== For autograder1 ==="
gradcheck_naive(lambda params: forward_backward_prop(data, labels, params), params)


# ## 3. Word2vec
# 
# *Please answer the third complementary problem before starting this part.*
# 
# In this part you will implement the `word2vec` models and train your own word vectors with stochastic gradient descent (SGD).

# In[ ]:

# Implement your skip-gram and CBOW models here

# Interface to the dataset for negative sampling
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0, 4)], [tokens[random.randint(0, 4)] for i in xrange(2 * C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext

def softmaxCostAndGradient(y1, target, w2):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector (\hat{r} in #
    #           the written component)                                #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       # 
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    out = np.dot(w2,y1)
    y2 = softmaxactual(out)
    cost = -math.log(y2[target])
    y2[target] -= 1.0
    d2= y2
    grad2 = np.dot(np.row_stack(d2),np.column_stack(y1))
    d1 = np.dot(np.transpose(w2), np.row_stack(d2))
    return cost, d1, grad2

def negSamplingCostAndGradient(y1, target, w2, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    #TODO Use negative sampling
    out = np.dot(w2,y1)
    y2 = softmaxactual(out)
    cost = -math.log(y2[target])
    y2[target] -= 1.0
    d2= y2
    grad2 = np.dot(np.row_stack(d2),np.column_stack(y1))
    d1 = np.dot(np.transpose(w2), np.row_stack(d2))
    return cost, d1, grad2

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #         
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of no more than 2*C strings, the context #
    #             words                                               #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    nextWordIndex = tokens[contextWords[(C+1)/2]]
    inputVector =inputVectors[tokens[currentWord]]
    predicted = np.dot(outputVectors, np.transpose(inputVector))
    cost, d1, gradOut = word2vecCostAndGradient(inputVector, nextWordIndex,outputVectors)
    y0 = np.zeros((np.shape(inputVectors)[0],1))
    y0[tokens[currentWord]][0] = 1.0
    grad1 = np.transpose(np.dot(d1, np.transpose(y0)))
    return cost, grad1, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient=softmaxCostAndGradient):
    """ CBOW model in word2vec """
    ###################################################################
    # Implement the continuous bag-of-words model in this function.   #         
    # Input/Output specifications: same as the skip-gram model        #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    nextWordIndex = tokens[contextWords[(C+1)/2]]
    inputVector =inputVectors[tokens[currentWord]]
    predicted = np.dot(outputVectors, np.transpose(inputVector))
    cost, d1, gradOut = word2vecCostAndGradient(inputVector, nextWordIndex,outputVectors)
    y0 = np.zeros((np.shape(inputVectors)[0],1))
    y0[tokens[currentWord]][0] = 1.0
    grad1 = np.transpose(np.dot(d1, np.transpose(y0)))
    return cost, grad1, gradOut

def sqr(x):
    return x*x
def normalizeRowsActual(x): 
    sumAll = math.sqrt(sum(map(sqr,x)))
    return x / sumAll
# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """
    return np.array(map(normalizeRowsActual, x))

# Test this function
print "=== For autograder ==="
print normalizeRows(np.array([[3.0, 4.0], [1, 2]]))  # the result should be [[0.6, 0.8], [0.4472, 0.8944]]

# Gradient check!
def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom
    return cost, grad

random.seed(31415)
np.random.seed(9265)
dummy_vectors = normalizeRows(np.random.randn(10, 3))
dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
print "==== Gradient check for skip-gram ===="
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
print "\n==== Gradient check for CBOW      ===="
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

print "\n=== For autograder ==="
print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :])
print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], negSamplingCostAndGradient)
print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :])
print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], negSamplingCostAndGradient)


# In[ ]:

# Now, implement SGD

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000

import glob
import os.path as op
import cPickle as pickle

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
            
    if st > 0:
        with open("saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None
    
def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent """
    ###################################################################
    # Implement the stochastic gradient descent method in this        #
    # function.                                                       #
    # Inputs:                                                         #
    #   - f: the function to optimize, it should take a single        #
    #        argument and yield two outputs, a cost and the gradient  #
    #        with respect to the arguments                            #
    #   - x0: the initial point to start SGD from                     #
    #   - step: the step size for SGD                                 #
    #   - iterations: total iterations to run SGD for                 #
    #   - postprocessing: postprocessing function for the parameters  #
    #        if necessary. In the case of word2vec we will need to    #
    #        normalize the word vectors to have unit length.          #
    #   - PRINT_EVERY: specifies every how many iterations to output  #
    # Output:                                                         #
    #   - x: the parameter value after SGD finishes                   #
    ###################################################################
    
    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
            
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    
    x = x0
    
    if not postprocessing:
        postprocessing = lambda x: x
    
    expcost = None
    
    for iter in xrange(start_iter + 1, iterations + 1):
        # ## YOUR CODE HERE
        # ## Don't forget to apply the postprocessing after every iteration!
        # ## You might want to print the progress every few iterations.
        
        # ## END YOUR CODE
        
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
            
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    
    return x


# **Show time! Now we are going to load some real data and train word vectors with everything you just implemented!**
# 
# We are going to use the Stanford Sentiment Treebank (SST) dataset to train word vectors, and later apply them to a simple sentiment analysis task.

# In[ ]:

# Load some data and initialize word vectors

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5


# In[ ]:

# Train word vectors (this could take a while!)

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / dimVectors,
                              np.zeros((nWords, dimVectors))), axis=0)
wordVectors0 = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient),
                   wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
# sanity check: cost at convergence should be around or below 10

# sum the input and output word vectors
wordVectors = (wordVectors0[:nWords, :] + wordVectors0[nWords:, :])

print "\n=== For autograder ==="
checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
checkIdx = [tokens[word] for word in checkWords]
checkVecs = wordVectors[checkIdx, :]
print checkVecs


# In[ ]:

# Visualize the word vectors you trained

_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords, :] + wordVectors0[nWords:, :])
visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying"]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2]) 

for i in xrange(len(visualizeWords)):
    plt.text(coord[i, 0], coord[i, 1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))
    
plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))


# ## 4. Sentiment Analysis
# 
# Now, with the word vectors you trained, we are going to perform a simple sentiment analysis.
# 
# For each sentence in the Stanford Sentiment Treebank dataset, we are going to use the average of all the word vectors in that sentence as its feature, and try to predict the sentiment level of the said sentence. The sentiment level of the phrases are represented as real values in the original dataset, here we'll just use five classes:
# 
#     "very negative", "negative", "neutral", "positive", "very positive"
#     
# which are represented by 0 to 4 in the code, respectively.
# 
# For this part, you will learn to train a softmax regressor with SGD, and perform train/dev validation to improve generalization of your regressor.

# In[ ]:

# Now, implement some helper functions

def getSentenceFeature(tokens, wordVectors, sentence):
    """ Obtain the sentence feature for sentiment analysis by averaging its word vectors """
    ###################################################################
    # Implement computation for the sentence features given a         #
    # sentence.                                                       #
    # Inputs:                                                         #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - wordVectors: word vectors for all tokens                    #
    #   - sentence: a list of words in the sentence of interest       #
    # Output:                                                         #
    #   - sentVector: feature vector for the sentence                 #
    ###################################################################
    
    sentVector = np.zeros((wordVectors.shape[1],))
    
    # ## YOUR CODE HERE
    
    # ## END YOUR CODE
    
    return sentVector

def softmaxRegression(features, labels, weights, regularization=0.0, nopredictions=False):
    """ Softmax Regression """
    ###################################################################
    # Implement softmax regression with weight regularization.        #
    # Inputs:                                                         #
    #   - features: feature vectors, each row is a feature vector     #
    #   - labels: labels corresponding to the feature vectors         #
    #   - weights: weights of the regressor                           #
    #   - regularization: L2 regularization constant                  #
    # Output:                                                         #
    #   - cost: cost of the regressor                                 #
    #   - grad: gradient of the regressor cost with respect to its    #
    #           weights                                               #
    #   - pred: label predictions of the regressor (you might find    #
    #           np.argmax helpful)                                    #
    ###################################################################
    
    prob = softmax(features.dot(weights))
    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1
    # A vectorized implementation of    1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2
    cost = np.sum(-np.log(prob[range(N), labels])) / N 
    cost += 0.5 * regularization * np.sum(weights ** 2)
    
    # ## YOUR CODE HERE: compute the gradients and predictions
    
    # ## END YOUR CODE
    
    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred

def precision(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size

def softmax_wrapper(features, labels, weights, regularization=0.0):
    cost, grad, _ = softmaxRegression(features, labels, weights, regularization)
    return cost, grad


# In[ ]:

# Gradient check always comes first
random.seed(314159)
np.random.seed(265)
dummy_weights = 0.1 * np.random.randn(dimVectors, 5)
dummy_features = np.zeros((10, dimVectors))
dummy_labels = np.zeros((10,), dtype=np.int32)    
for i in xrange(10):
    words, dummy_labels[i] = dataset.getRandomTrainSentence()
    dummy_features[i, :] = getSentenceFeature(tokens, wordVectors, words)
print "==== Gradient check for softmax regression ===="
gradcheck_naive(lambda weights: softmaxRegression(dummy_features, dummy_labels, weights, 1.0, nopredictions=True), dummy_weights)

print "\n=== For autograder ==="
print softmaxRegression(dummy_features, dummy_labels, dummy_weights, 1.0)


# In[ ]:

# Try different regularizations and pick the best!

# ## YOUR CODE HERE

regularization = 0.0  # try 0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01 and pick the best

# ## END YOUR CODE

random.seed(3141)
np.random.seed(59265)
weights = np.random.randn(dimVectors, 5)

trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)

for i in xrange(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
# We will do batch optimization
weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels, weights, regularization), weights, 3.0, 10000, PRINT_EVERY=100)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)

for i in xrange(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
_, _, pred = softmaxRegression(devFeatures, devLabels, weights)
print "Dev precision (%%): %f" % precision(devLabels, pred)


# In[ ]:

# Write down the best regularization and accuracy you found
# sanity check: your accuracy should be around or above 30%

# ## YOUR CODE HERE

BEST_REGULARIZATION = 1
BEST_ACCURACY = 0.0

# ## END YOUR CODE

print "=== For autograder ===\n%g\t%g" % (BEST_REGULARIZATION, BEST_ACCURACY)


# In[ ]:

# Test your findings on the test set

testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)

for i in xrange(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
_, _, pred = softmaxRegression(testFeatures, testLabels, weights)
print "=== For autograder ===\nTest precision (%%): %f" % precision(testLabels, pred)


# #### Extra Credit
# 
# Train your own classifier for sentiment analysis! We will not provide any starter code for this part, but you can feel free to reuse the code you've written before, or write some new code for this task. Also feel free to refer to the code we provided you with to see how we scaffolded training for you.
# 
# Try to contain all of your code in one code block. You could start by using multiple blocks, then paste code together and remove unnecessary blocks. Report, as the last two lines of the output of your block, the dev set accuracy and test set accuracy you achieved, in the format we used above.
# 
# *Note: no credits will be given for this part if you use the dev or test sets for training, or if you fine-tune your regularization or other hyperparameters on the test set.*

# In[ ]:

# ## YOUR CODE HERE

# ## END YOU CODE

'''
_, _, pred = softmaxRegression(devFeatures, devLabels, weights)
print "=== For autograder ===\nDev precision (%%): %f" % precision(devLabels, pred)
_, _, pred = softmaxRegression(testFeatures, testLabels, weights)
print "Test precision (%%): %f" % precision(testLabels, pred)
'''