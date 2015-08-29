#coding=utf-8
import MySQLdb
import math
import numpy

'''
try:
    conn=MySQLdb.connect(host='localhost',user='root',passwd='scut2428',db='hushen300',port=3306)
    cur=conn.cursor()
    a='2015-05-11 15:00:00 '
    x='2015-06-11 15:00:00 '
    #cur.execute("select * from `000001.sz` where DATEDIFF(date_time,'2015-05-12')")
    b=cur.execute("select * from `000001.sz` where  date_time>'2015-06-10'")
    #b=cur.fetchall()
    print b
    conn.commit()
    cur.close()
    conn.close()
except MySQLdb.Error,e:
    print "Mysql Error %d: %s" % (e.args[0], e.args[1])
'''
'''
try:
    conn=MySQLdb.connect(host='localhost',user='root',passwd='scut2428',db='hushen300',port=3306)
    cur=conn.cursor()
    a=cur.execute('select distinct code from stock_list ')
    b=cur.fetchall()
    print b
    cur.execute('delete from stock_list')
    sql='insert into stock_list values(%s)'
    cur.executemany(sql,b)

    conn.commit()
    cur.close()
    conn.close()
except MySQLdb.Error,e:
    print "Mysql Error %d: %s" % (e.args[0], e.args[1])
'''
#得出的是所有的天数数据矩阵245*435600

try:
    conn=MySQLdb.connect(host='localhost',user='root',passwd='scut2428',db='hushen300',port=3306)
    cur=conn.cursor()
    c=cur.execute("select * from stock_list")
    #c='`000001.SZ`'
    print c
    #cur.execute("select * from %s" %str.lower(c))
    a=cur.fetchall()
    print type(a[0])
    #print str(list(a[0]))
    #c=str.lower(str(a[0]))
    print a
    #print str.lower(str(a))

    changetime=['2014-06-16','2014-12-15','2015-01-26','2015-05-21','2015-06-15']
    x='2014-06-17'
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
                    print nun
                #print len(data)
            except:
                print 'fail'
        print len(data)
        stocklist.append(data)
        print nun

    print len(stocklist)
    print stocklist[0][0][0]
    print stocklist[1][0][0]
    print len(stocklist[0])
    print len(stocklist[0][0])
    print len(stocklist[0][0][0])

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
print len(a)
print len(a[0])
print a[0][0]
print a[0][1]
print a[244][0]
print a[244][1]
#print a[0]
a=numpy.array(list(a))
a=a.reshape(len(a),len(a[0])*len(a[0][0]))
print a[0][0]
print a[0][1]
print a[1][0]
print a[-1][-1]
print a.shape
train_set_x=a[0:220]
print train_set_x.shape
'''
#得出target数据
try:
    conn=MySQLdb.connect(host='localhost',user='root',passwd='scut2428',db='hushen300',port=3306)
    cur=conn.cursor()
    c=cur.execute("select * from `if.cfe`")
    #c='`000001.SZ`'
    print c
    #cur.execute("select * from %s" %str.lower(c))
    a=cur.fetchall()
    print type(a[0])
    #print str(list(a[0]))
    #c=str.lower(str(a[0]))
    y=[]
    target=0
    #print str.lower(str(a))
    for i in range(len(a)):
        if(a[i][0]>a[i][3]):
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
'''