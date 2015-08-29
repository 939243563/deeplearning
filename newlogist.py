#coding=utf-8
import MySQLdb
import numpy
import theano
import theano.tensor as T
rng= numpy.random
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
        set=[]
        for n in ['open','high','low','close','volume']:
            cur.execute("select "+n+" from %s"%stockname)
            a=cur.fetchall()
            #a=str(a)
            #print type(a)
            set.append(a)

            #print a
        #print set[0][1]

        b.append(set)
    print len(b[230][1])
    print len(b[0][0])
    train_set_x_x=[]
    #data_set 表示的是３００只股票所有数据　data_set[0]表示３００只股票的open数据
    data_set=[]
    for j in range(len(b[0])):
        data=[]
        #print type(data)
        for index in range(len(b)):
            data.append(b[index][j])
            #data[index]=list(data[index])
            #print type(data[index])
        #data=numpy.matrix(data)
        #print type(data)
        #print data.shape
        data_set.append(data)
    #data_set[0][0]=numpy.array(data_set[0][0])
    #data_set[0]=numpy.asarray(list(data_set[0]))
    #print data_set[0].shape[1]
    #print len(data_set[0])
    #print type(data_set[0])
    #print list(data_set[0]).shape()
    #print type(data_set[0][0])
    #print type(data_set[0][0][0])
    #print len(data_set[0])
    all_data_set=numpy.array(data_set)
    #print all_data_set.shape
    print len(all_data_set)
    print all_data_set[0][0][0]
    #print (data_set[4][1][1])
    # Load the dataset
    #10天的数据作为训练，递推１天
    train_set_x=[]
    test_set_x=[]
    valid_set_x=[]
    for i in range(len(all_data_set)):
        k=0
        while k<41:
            train_set_x.append(all_data_set[i][0:,k*242:k*242+10*242])
            k=k+1
        while k<48:
            test_set_x.append(all_data_set[i][0:,k*242:k*242+10*242])
            k=k+1
        while k<55:
            valid_set_x.append(all_data_set[i][0:,k*242:k*242+10*242])
            k=k+1
    temp_train_set=numpy.array(train_set_x)
    temp_test_set=numpy.array(test_set_x)
    temp_valid_set=numpy.array(valid_set_x)
    train_set_x=[]
    test_set_x=[]
    valid_set_x=[]
    print len(temp_train_set)
    #将４维的训练集变为２维
    for i in temp_train_set:
        g=[]
        for j in i:
            for k in j:
                g.extend(k)

        train_set_x.append(g)
    for i in temp_test_set:
        g=[]
        for j in i:
            for k in j:
                g.extend(k)

        test_set_x.append(g)
    for i in temp_valid_set:
        g=[]
        for j in i:
            for k in j:
                g.extend(k)
        valid_set_x.append(g)
    #对训练集进行矩阵化
    train_set_x=numpy.array(train_set_x)
    test_set_x=numpy.array(test_set_x)
    valid_set_x=numpy.array(valid_set_x)
    #train_set_x=numpy.array(train_set_x)
    #print train_set_x.shape
    #print len(test_set_x[0])
    #print len(valid_set_x)

    #取target
    cur.execute("select * from IF_CFE")
    data_set_y=cur.fetchall()
    all_data_set_y=[]
    num=0
    print len(data_set_y)

    while num+2991<len(data_set_y):
        if(data_set_y[num+2720][0]>data_set_y[num+2991][3]):
            target=0
        else:
            target=1
        all_data_set_y.append(target)
        num=num+272
    print all_data_set_y
    temp=[]
    temp.extend(all_data_set_y)
    #同一时间段的不同训练数据有着同一个target
    for l in range(4):
        all_data_set_y.extend(temp)
    print len(all_data_set_y)
    print all_data_set_y

    train_set_y=[]
    test_set_y=[]
    valid_set_y=[]
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
    train_set=[train_set_x,train_set_y]
    test_set=[test_set_x,test_set_y]
    valid_set=[valid_set_x,valid_set_y]
    #train_set=numpy.array(train_set)
    #test_set=numpy.array(test_set)
    #valid_set=numpy.array(valid_set)
    print type(train_set_x)
    print type(test_set)
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

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

dataset=load_data()
N= 205
feats= 300*242*10
D= (dataset[0][0], dataset[0][1])
training_steps= 10000

#Declare Theano symbolic variables
x= T.matrix("x")
y= T.vector("y")
w= theano.shared(rng.randn(feats), name="w")
b= theano.shared(0., name="b")
print "Initial model:"
print w.get_value(), b.get_value()

#Construct Theano expression graph
p_1= 1 / (1 + T.exp(-T.dot(x, w) - b))   #Probability that target = 1
prediction= p_1 > 0.5                    # Theprediction thresholded
xent= -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost= xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw,gb = T.grad(cost, [w, b])             #Compute the gradient of the cost
                                          # (we shall return to this in a
                                          #following section of this tutorial)


#Compile
train= theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b -0.1 * gb)))
predict= theano.function(inputs=[x], outputs=prediction)

#Train
for i in range(training_steps):
    pred,err = train(D[0], D[1])
print "Final model:"
print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])