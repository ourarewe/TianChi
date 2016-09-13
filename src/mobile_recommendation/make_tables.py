#coding:utf-8
import numpy as np
from numpy import *
import pandas as pd
import mysql.connector

date_list = pd.date_range(start='2014-11-18', end='2014-11-18', freq='1d')


cnx = mysql.connector.connect(user='root', password='555321',
                              host='127.0.0.1',
                              database='tianchi')
cursor = cnx.cursor()

for day in date_list:
    day_str = str(day)[0:10]
    print day_str
    query=("insert into tianchi_fresh_comp_train_P_R \
            select * from ( \
            select user_id,item_id,time \
            ,sum(if(behavior_type=1,1,0)) browse \
            ,sum(if(behavior_type=2,1,0)) collect \
            ,sum(if(behavior_type=3,1,0)) addCart \
            ,sum(if(behavior_type=4,1,0)) buy \
            from tianchi_fresh_comp_train_P \
            where time like '%s \%' \
            group by user_id,item_id)t;")
    print query
    cursor.execute(query,(day_str))
    print ">>>>>>>>>>>"
    #query("drop table if exists test;")
    #cursor.execute(query)
    
cnx.commit()
cursor.close()
cnx.close()

print 'finished'