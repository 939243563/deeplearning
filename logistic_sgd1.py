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
import pylab

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
    try:
        conn=MySQLdb.connect(host='localhost',user='root237',passwd='root237',db='test',port=3306)
        cur=conn.cursor()
        c=cur.execute("select * from stock_list")
        #c='`000001.SZ`'
        #print c
        #cur.execute("select * from %s" %str.lower(c))
        a=cur.fetchall()
        print type(a)
        #a=numpy.array(a)%256
        print type(a[0])
        #print str(list(a[0]))
        #c=str.lower(str(a[0]))
        #print a
        #print str.lower(str(a))

        changetime=['2014-06-16','2014-12-15','2015-01-26','2015-05-21','2015-06-15']
        #x='2014-06-17'
        stocklist=[]
        nun=0
        #stocklist包括了４个周期内的所有数据,stcoklist[0]表示的是第一个周期３００只股票的数据，stocklist[0][0]表示第一只股票的数据
        #stocklist[0][0][0]表示的是股票每分钟的６个数据
        for j in range(len(changetime)-1):
            data=[]

            for i in a:
                try:
                    #b=cur.execute("select * from `%s`" %str.lower(list(i)[0]))
                    b=cur.execute("select open,high,low,close,volume,amt from `%s`"%str.lower(list(i)[0])+"where date_time < %s and date_time > %s",[changetime[j+1],changetime[j]])
                    #data.append(cur.fetchall())
                    if (b!=0):
                        data.append(cur.fetchall())

                        nun+=1
                        #print nun
                    #print len(data)
                except:
                    print 'fail'
            print len(data)
            stocklist.append(data)
            print nun
        '''
        print len(stocklist)
        print stocklist[0][0][0]
        print stocklist[1][0][0]
        print len(stocklist[0])
        print len(stocklist[0][0])
        print len(stocklist[0][0][0])
        '''
        cur.execute("select * from `if.cfe`")
        #c='`000001.SZ`'
        #print c
        #cur.execute("select * from %s" %str.lower(c))
        m=cur.fetchall()
        print type(m[0])
        #print str(list(a[0]))
        #c=str.lower(str(a[0]))
        y=[]
        target=0
        #print str.lower(str(a))
        for i in range(len(m)):
            if(m[i][0]>m[i][3]):
                target=0
                y.append(target)
            else:
                target=1
                y.append(target)
        print len(y)
        print y
        conn.commit()
        cur.close()
        conn.close()

    except MySQLdb.Error,e:
        print "Mysql Error %d: %s" % (e.args[0], e.args[1])
    #5天为一个训练数据
    day0=len(stocklist[0][0])/242
    day1=len(stocklist[1][0])/242
    day2=len(stocklist[2][0])/242
    day3=len(stocklist[3][0])/242
    day=day0+day1+day2+day3
    print day0,day1,day2,day3,day
    #a[0]表示的是第一天的３００只股票的数据
    a=[]
    for i in range(day0):
        b=[]
        for j in range(300):
            b.extend(stocklist[0][j][i*242:(i+1)*242])
        a.append(b)
    for i in range(day1):
        b=[]
        for j in range(300):
            b.extend(stocklist[1][j][i*242:(i+1)*242])
        a.append(b)
    for i in range(day2):
        b=[]
        for j in range(300):
            b.extend(stocklist[2][j][i*242:(i+1)*242])
        a.append(b)
    for i in range(day3):
        b=[]

        for j in range(300):
            b.extend(stocklist[3][j][i*242:(i+1)*242])
        a.append(b)
    length=len(a)
    '''
    print len(a)
    print len(a[0])
    print a[0][0]
    print a[0][1]
    print a[244][0]
    print a[244][1]
    '''
    #print a[0]
    a=numpy.array(list(a))
    a=a.reshape(len(a),len(a[0])*len(a[0][0]))
    '''
    print a[0][0]
    print a[0][1]
    print a[1][0]
    print a[-1][-1]
    print a.shape
    '''
    set_x=[]

    print len(a)
    for i in range(length-5):
        set_x.append(a[i:i+5])
    set_x=numpy.array(set_x)
    set_x=set_x.reshape(len(set_x),len(set_x[0])*len(set_x[0][0]))
    print set_x.shape
    #set_x=set_x%256
    train_set_x=set_x[0:150]
    valid_set_x=set_x[200:240]
    test_set_x=set_x[150:200]
    train_set_y=y[6:156]
    valid_set_y=y[206:246]
    test_set_y=y[156:206]
    print test_set_y
    print train_set_x.shape
    print train_set_x[0]
    '''
    pylab.subplot(2,1,1)
    a=train_set_x[0]
    a=a.reshape(1,1500,1452).swapaxes(1,2)
    print type(a[0])
    print a.shape,'fff'
    pylab.imshow(a[0,:,:])
    pylab.show()
    '''
    '''
    train_set_x=a[0:220]
    valid_set_x=a[240:245]
    test_set_x=a[220:240]
    train_set_y=y[1:221]
    valid_set_y=y[241:246]
    test_set_y=y[221:241]
    '''
    print test_set_y



    train_set=[train_set_x,train_set_y]
    test_set=[test_set_x,test_set_y]
    valid_set=[valid_set_x,valid_set_y]
    #train_set=numpy.array(train_set)
    #test_set=numpy.array(test_set)
    #valid_set=numpy.array(valid_set)


    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.



    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    #print train_set_x,type(train_set_y)
    #print test_set_y[0:10].eval()
    #print test_set_y.__getstate__()
    rval = [(train_set_x, train_set_y),(valid_set_x,valid_set_y),(test_set_x, test_set_y)]
    return rval

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

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1,
                           batch_size=10):
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
        user='root237',
        passwd='root237',
        db ='hushen300',
        )
    cur=conn.cursor()
    cur.execute("select * from `if.cfe`")
    data1=cur.fetchall()
    profit=[]
    for i in range(80):     #做多
            profit.append(float(data1[166+i][3])-float(data1[166+i][0]))
    print profit
    conn.commit()
    cur.close()
    conn.close()
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
    classifier = LogisticRegression(input=x, n_in=5*6*300*242, n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    datasets = load_data()


    train_set_x, train_set_y = shared_dataset(datasets[0])
    print train_set_x.get_value().shape
    #print train_set_x.get_value()

    valid_set_x, valid_set_y = shared_dataset(datasets[1])

    test_set_x, test_set_y = shared_dataset(datasets[2])
    #print test_set_y

    # compute number of minibatches for training, validation and testing
    n_train_batches = (train_set_x.get_value(borrow=True).shape[0]-150) / batch_size+1
    print n_train_batches
    n_valid_batches = (valid_set_x.get_value(borrow=True).shape[0])/ batch_size
    n_test_batches = (test_set_x.get_value(borrow=True).shape[0]) / batch_size

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            #x: test_set_x,
            #y: test_set_y
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            #x: valid_set_x,
            #y: valid_set_y
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    pred_model=theano.function(
        inputs=[index],
        outputs=classifier.pred(),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
            #x: test_set_x
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
            x: train_set_x[index * batch_size: index * batch_size+150],
            y: train_set_y[index * batch_size: index * batch_size+150]
            #x: train_set_x,
            #y: train_set_y
            #x: train_set_x[0: 220],
            #y: train_set_y[0: 220]
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
    while (epoch < 8) and (not done_looping):



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
    pred_data=pred_data.reshape(1,80).tolist()[0]
    profit=[]

    for i in range(len(pred_data)):
        if(pred_data[i]==1):     #做多
            profit.append(float(data1[156+i][3])-float(data1[156+i][0]))
        if(pred_data[i]==0):     #做空
            profit.append(float(data1[156+i][0])-float(data1[156+i][3]))
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
