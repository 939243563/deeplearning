#coding=utf-8
"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import os
import sys
import time
import MySQLdb
import numpy

import theano
import theano.tensor as T
import matplotlib.pyplot as plt

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
    def pred(self):
        return self.y_pred
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    conn= MySQLdb.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='scut2428',
        db ='data',
        )
    cur=conn.cursor()
    cur.execute("select * from stockList")
    stock=cur.fetchall()
    print len(stock)
    #b[]表示所有股票的所有数据b[0]代表第一个股票的数据
    b=[]
    for stockname in stock:
        #print type(stockname)
        stockname=''.join(stockname).split('.')
        #print stockname
        stockname='stock'+stockname[0]+'_'+stockname[-1]
        #set[]:表示的是每个股票各个属性的所有数据　set[0]代表某个股票的open数据


        cur.execute("select open,high,low,close,volume from %s"%stockname)
        a=cur.fetchall()
            #a=str(a)
            #print type(a)


            #print a
        #print set[0][1]

        b.append(a)

    data_set_x=[]
    train_set_x=[]
    test_set_x=[]
    valid_set_x=[]
    for i in range(64):
        data=[]
        for j in range(len(b)):
            data.extend(b[j][i*242:(i+1)*242])
        data_set_x.append(data)
    train_set_x=data_set_x[0:60]
    valid_set_x=data_set_x[8:62]
    test_set_x=data_set_x[10:64]
    print len(train_set_x)
    print len(train_set_x[0])
    temp_train_set=numpy.array(train_set_x)
    temp_test_set=numpy.array(test_set_x)
    temp_valid_set=numpy.array(valid_set_x)
    train_set_x=[]
    test_set_x=[]
    valid_set_x=[]
    print len(temp_train_set)
    print len(temp_test_set)
    #将４维的训练集变为２维
    for i in temp_train_set:
        g=[]
        for j in i:

            g.extend(j)

        train_set_x.append(g)
    for i in temp_test_set:
        g=[]
        for j in i:

            g.extend(j)

        test_set_x.append(g)

    for i in temp_valid_set:
        g=[]
        for j in i:

            g.extend(j)
        valid_set_x.append(g)



    '''
    for i in range(len(all_data_set)):
        k=0
        while k<41:
            train_set_x.append(all_data_set[i][0:,k*242:(k+1)*242])
            k=k+1
        while k<48:
            test_set_x.append(all_data_set[i][0:,k*242:k*242+10*242])
            k=k+1
        while k<55:
            valid_set_x.append(all_data_set[i][0:,k*242:k*242+10*242])
            k=k+1
    '''




    #对训练集进行矩阵化
    train_set_x=numpy.array(train_set_x)
    test_set_x=numpy.array(test_set_x)
    valid_set_x=numpy.array(valid_set_x)
    #train_set_x=numpy.array(train_set_x)
    #print train_set_x.shape
    #print len(test_set_x[0])
    #print len(valid_set_x)
    print train_set_x[0][1]
    print len(train_set_x[0])
    print test_set_x[0][1]
    print len(test_set_x[0])
    print len(valid_set_x[0])
    #取target
    cur.execute("select * from IF_CFE")
    data_set_y=cur.fetchall()
    all_data_set_y=[]
    num=0
    print len(data_set_y)

    while num+543<len(data_set_y):
        if(data_set_y[num+272][0]>data_set_y[num+543][3]):
            target=0
        else:
            target=1
        all_data_set_y.append(target)
        num=num+272
    print len(all_data_set_y)
    print all_data_set_y[0]
    '''
    temp=[]
    temp.extend(all_data_set_y)
    #同一时间段的不同训练数据有着同一个target
    for l in range(4):
        all_data_set_y.extend(temp)
    print len(all_data_set_y)
    print all_data_set_y
    '''
    train_set_y=[]
    test_set_y=[]
    valid_set_y=[]
    train_set_y=all_data_set_y[0:60]
    valid_set_y=all_data_set_y[8:62]
    test_set_y=all_data_set_y[10:64]

    print len(test_set_y)
    print len(train_set_y)
    print test_set_y
    '''
    for l in range(5):
        train_set_y.extend(all_data_set_y[l*55:41+l*55])
        test_set_y.extend(all_data_set_y[41+l*55:48+l*55])
        valid_set_y.extend(all_data_set_y[48+l*55:55+l*55])
    #print train_set_y
    #print len(train_set_y)
    #print len(test_set_y)
    #print len(valid_set_y)
    #print valid_set_y
    #train_set_x=numpy.array(train_set_x)
    #test_set_x=numpy.array(test_set_x)
    #valid_set_x=numpy.array(valid_set_x)
    #train_set_y=numpy.array(train_set_y)
    #test_set_y=numpy.array(test_set_y)
    #valid_set_y=numpy.array(valid_set_y)
    print type(train_set_x),'a'
    print type(train_set_y),'b'
    '''
    train_set=[train_set_x,train_set_y]
    test_set=[test_set_x,test_set_y]
    valid_set=[valid_set_x,valid_set_y]
    #train_set=numpy.array(train_set)
    #test_set=numpy.array(test_set)
    #valid_set=numpy.array(valid_set)
    print type(train_set_x)
    print type(test_set)
    print test_set[1],'ad'
    '''

    all_data_set_x=[]

    cur.execute("select * from stock000001_SZ")
    set_x=cur.fetchall()
    print set_x
    all_data_set_x.extend(set_x)
    all_data_set_y=[]
    n=0
    while n<len(all_data_set_x):
        day=n/242
        if(all_data_set_x[n][3]<all_data_set_x[(day+1)*242-1][3]):
            target=[0]
            all_data_set_y.extend(target)
        else:
            target=[1]
            all_data_set_y.extend(target)

        n=n+1


    train_set_x=all_data_set_x[epoch*5*242:15*242+epoch*5*242]
    test_set_x=all_data_set_x[15*242+epoch*5*242:18*242+epoch*5*242]
    valid_set_x=all_data_set_x[18*242+epoch*5*242:20*242+epoch*5*242]
    train_set_y=all_data_set_y[epoch*20*242:46*242+epoch*20*242]
    test_set_y=all_data_set_y[46*242+epoch*20*242:56*242+epoch*20*242]
    valid_set_y=all_data_set_y[56*242+epoch*20*242:66*242+epoch*20*242]

    train_set=[train_set_x,train_set_y]
    test_set=[test_set_x,test_set_y]
    valid_set=[valid_set_x,valid_set_y]
    '''
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    print train_set_x,type(train_set_y)

    rval = [(train_set_x, train_set_y),(valid_set_x,valid_set_y),(test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=20,
                           batch_size=2):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    conn= MySQLdb.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='scut2428',
        db ='data',
        )
    cur=conn.cursor()
    cur.execute("select * from IF_CFE")
    data=cur.fetchall()

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=5*300*242, n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    datasets = load_data()


    train_set_x, train_set_y = datasets[0]
    print train_set_x.get_value().shape
    #print train_set_x.get_value()

    valid_set_x, valid_set_y = datasets[2]

    test_set_x, test_set_y = datasets[1]
    #print test_set_y

    # compute number of minibatches for training, validation and testing
    n_train_batches = (train_set_x.get_value(borrow=True).shape[0]-8) / batch_size+1
    print n_train_batches
    n_valid_batches = (valid_set_x.get_value(borrow=True).shape[0])/ batch_size
    n_test_batches = (test_set_x.get_value(borrow=True).shape[0]) / batch_size

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    pred_model=theano.function(
        inputs=[index],
        outputs=classifier.pred(),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            #y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    #g_W,g_b = T.grad(cost=cost,wrt=[classifier.W,classifier.b])
    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: index * batch_size+8],
            y: train_set_y[index * batch_size: index * batch_size+8]
        }
    )
    #classify = theano.function(inputs=[x], outputs=classifier.y_pred)
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    color=['r','g']
    pred_data=[]
    while (epoch < n_train_batches) and (not done_looping):



        #for minibatch_index in xrange(n_train_batches):
        #for minibatch_index in xrange(1):


        minibatch_avg_cost = train_model(epoch)

        print classifier.W.get_value()

        #print minibatch_avg_cost
        predition=[pred_model(epoch)]
        #print predition
        pred_data.extend(predition)
        epoch = epoch + 1
        #print classifier.W.get_value()
        # iteration number
        iter = (epoch - 1) * n_train_batches
        #print minibatch_index + 1
        #print validation_frequency
        if (iter) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)

            #print minibatch_index + 1
            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    epoch + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )
            )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                # test it on the test set

                test_losses = [test_model(i)
                               for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)

                print(
                    (
                        '     epoch %i, minibatch %i/%i, test error of'
                        ' best model %f %%'
                    ) %
                    (
                        epoch,
                        epoch + 1,
                        n_train_batches,
                        test_score * 100.
                    )
                )

            #print classifier.y_pred

    print len(pred_data)
    pred_data=numpy.array(pred_data)
    pred_data=pred_data.reshape(1,54).tolist()[0]
    profit=[]

    for i in range(len(pred_data)):
        if(pred_data[i]==1):     #做多
            profit.append(data[272*(11+i+1)-2][3]-data[272*(11+i)][0])
        if(pred_data[i]==0):     #做空
            profit.append(data[272*(11+i)][0]-data[272*(11+i+1)-2][3])
    income=[]
    print profit
    num=0.0

    m=0
    for i in profit:
        if (i>0):
            num+=1
        m=m+i
        income.append(m)
    print income
    print len(pred_data)
    x=[]
    win_rate=num/len(profit)
    print len(profit)
    print ('胜率为：%f %%' % (win_rate*100))
    for i in range(len(income)):
        x.append(i)
    plt.plot(x,income,'b*')
    plt.plot(x,income)

    end_time = time.clock()

    #print len(pred_data)


    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    plt.show()
if __name__ == '__main__':
    sgd_optimization_mnist()
