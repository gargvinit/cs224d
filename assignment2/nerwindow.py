from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        self.sparams.L = wv.copy() # store own representations
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)

    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        xf = []
        for idx in window:
            xf.extend( self.sparams.L[idx]) # extract representation
        tanhX = tanh(self.params.W.dot(xf) + self.params.b1)
        softmaxP = softmax(self.params.U.dot(tanhX) + self.params.b2)
        y = make_onehot(label, len(softmaxP))
        delta2 = softmaxP -y
        self.grads.U += outer(delta2, tanhX) + self.lreg * self.params.U
        self.grads.b2 += delta2
        delta1 = self.params.U.T.dot(delta2)*(1. - tanhX*tanhX)
        self.grads.W += outer(delta1, xf) + self.lreg * self.params.W
        self.grads.b1 += delta1
        #for xw in window:
            #self.sgrads.L[xw] = self.params.W.T.dot(delta1)/len(window)


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]
        #TODO Use windows as transposed
        xu = [self.sparams.L[idx] for idx in windows]# extract representation
        xf =reduce(lambda x,y: x.extend(y),xu)

        print self.params.W.shape
        tanhX = tanh(self.params.W.dot(xf) + self.params.b1)
        softmaxP = softmax(self.params.U.dot(tanhX) + self.params.b2)
        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """
        P = self.predict_proba(windows)
        return argmax(P, axis=1)


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """
      
        '''if not hasattr(windows[0], "__iter__"):
            windows = [windows]
        x = []
        for window in windows:'''
        xf = []
        for idx in windows:
            xf.extend( self.sparams.L[idx]) # extract representation
        tanhX = tanh(self.params.W.dot(xf) + self.params.b1)
        softmaxP = softmax(self.params.U.dot(tanhX) + self.params.b2)
        J = -1*log(softmaxP[labels]) # cross-entropy loss
        Jreg = (self.lreg / 2.0) *( sum(self.params.W**2.0)+ sum(self.params.U**2.0))
        return J + Jreg
    
import sys, os
from numpy import *
from matplotlib.pyplot import *
from misc import random_weight_matrix
random.seed(10)
print random_weight_matrix(3,5)
from nerwindow import WindowMLP
    
import data_utils.utils as du
import data_utils.ner as ner
# Load the starter word vectors
wv, word_to_num, num_to_word = ner.load_wv('data/ner/vocab.txt',
                                           'data/ner/wordVectors.txt')
tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
num_to_tag = dict(enumerate(tagnames))
tag_to_num = du.invert_dict(num_to_tag)

# Set window size
windowsize = 3

# Load the training set
docs = du.load_dataset('data/ner/train')
X_train, y_train = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                      wsize=windowsize)

# Load the dev set (for tuning hyperparameters)
docs = du.load_dataset('data/ner/dev')
X_dev, y_dev = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                  wsize=windowsize)

# Load the test set (dummy labels only)
docs = du.load_dataset('data/ner/test.masked')
X_test, y_test = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                    wsize=windowsize)
clf = WindowMLP(wv, windowsize=windowsize, dims=[None, 100, 5],
                reg=0.001, alpha=0.01)
clf.grad_check(X_train[0], y_train[0])
clf.train_sgd( X_train, y_train)