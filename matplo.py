#coding=utf-8
import matplotlib.pyplot as plt
from pylab import *
c=[1,2,3]
d=[2,3,4]
a=open('test1.txt','w')
a.write(str((c,d)))
a.close()
x=[1,2,3]
y=[2,3,4]
a=[]
i=1
j=0
a.append((i,j))
print a
temp=2
for i in range(9):
    x[0]+=temp
    print x
    subplot(3,3,i+1)
    plt.plot(x,y)
    plt.title(i)
plt.show()
