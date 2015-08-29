#coding=utf-8
'''
from theano import function, config, shared, sandbox, tensor, Out
import numpy
import time
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import numpy as np
#for idx,color in enumerate("rgbyck"):
    #plt.subplot(231+idx,axisbg=color)
plt.figure(1)
plt.figure(2)
ax1=plt.subplot(211)
ax2=plt.subplot(212)
x=np.linspace(0,3,100)
for i in range(5):
    plt.figure(1)
    plt.plot(x,np.exp(i*x/3))
    plt.sca(ax1)
    plt.plot(x,np.sin(i*x))
    plt.sca(ax2)
    plt.plot(x,np.cos(i*x))
plt.figure(3)
plt.plot([1,2],[1,3])
title('sh')
xlabel('sfaf')
ylabel('y')
savefig('demo.jpg')
import matplotlib.pyplot as plt

# 1D data
x = [1,2,3,4,5]
y = [2.3,3.4,1.2,6.6,7.0]

plt.figure(figsize=(12,6))

plt.subplot(231)
plt.plot(x,y)
plt.title("plot")

plt.subplot(232)
plt.scatter(x, y)
plt.title("scatter")

plt.subplot(233)
plt.pie(y)
plt.title("pie")

plt.subplot(234)
plt.bar(x, y)
plt.title("bar")

# 2D data
import numpy as np
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z    = Y**2 + X**2

plt.subplot(235)
plt.contour(X,Y,Z)
plt.colorbar()
plt.title("contour")




plt.show()
'''
"""
import MySQLdb
import numpy
conn=MySQLdb.connect(host='localhost',user='root237',passwd='root237',db='test',port=3306)
#conn1=MySQLdb.connect(host='localhost',user='root237',passwd='root237',db='test',port=3306)
cur=conn.cursor()
#co=conn1.cursor()
d=cur.execute("select distinct * from stock_list")
print d
c=cur.fetchall()
print c
n=0
for i in c:
    '''
    try:
        cur.execute("create table temp as (select * from `%s` order by date_time asc)"%str.lower(list(i)[0]))
        cur.execute("drop table `%s`"%str.lower(list(i)[0]))
        cur.execute("alter table temp rename to `%s`"%str.lower(list(i)[0]))
        conn.commit()
        n+=1
    except:
        print 'fail'
'''
    try:
        #cur.execute("select max(date_time) from `%s`"%str.lower(list(i)[0]))
        #cur.execute("alter table `%s` add primary key(date_time)"%str.lower(list(i)[0]))
        cur.execute("alter table `%s` modify open double "%str.lower(list(i)[0]))
        cur.execute("alter table `%s` modify high double "%str.lower(list(i)[0]))
        cur.execute("alter table `%s` modify low double "%str.lower(list(i)[0]))
        cur.execute("alter table `%s` modify close double "%str.lower(list(i)[0]))
        cur.execute("alter table `%s` modify volume double "%str.lower(list(i)[0]))
        cur.execute("alter table `%s` modify amt double "%str.lower(list(i)[0]))
        conn.commit()
        #b=cur.fetchall()
        #print b,i,'a'
        '''
        co.execute("select * from `%s`"%str.lower(list(i)[0]))
        a=co.fetchall()
        a=numpy.array(a).tolist()
        #print a
        print len(a)
        sql="insert into `%s`"%str.lower(list(i)[0])+"(open,high,low,close,volume,amt,date_time) values(%s,%s,%s,%s,%s,%s,%s)"
        print sql
        cur.executemany(sql,a)
        conn.commit()
        '''
        n+=1

    except:
        print 'fail'



print n,'n'
#conn1.commit()

#co.close()
cur.close()
"""

import MySQLdb
conn=MySQLdb.connect(host='localhost',user='root237',passwd='root237',db='hushen300',port=3306)
cur=conn.cursor()




















