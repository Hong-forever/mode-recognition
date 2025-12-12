#! /usr/bin/env python
#coding=utf-8
from __future__ import division
import copy
import sys

path=sys.argv[1]
reader=open(path,'rb')
lines_1=reader.read().split('\n')
lines=[]
for line in lines_1:
    line_temp=copy.deepcopy(line)
    if len(line_temp.strip())<>0:
       lines.append(line)
reader.close()
output=open(r'check.txt','w')
right=True
if len(lines)<>2500:
    print 'row count error.'
    
    right=False
for i,line in enumerate(lines):
    str_list=line.split()
    if len(str_list)<>4:
        print 'column count error at row %d' %i
        exit()
        
        right=False
    try:
        id=int(str_list[1])
    except:
        print '2nd column at row %d must be interger' %i
        
        right=False
    try:
        sample_id=int(str_list[2])
        if(sample_id>=2500 or sample_id <0):
            print 'id number error at row %d' %i
          
            right=False
    except:
        print '3rd column at row %d must be interger' %i
       
        right=False
    if str_list[3]<>'negative' and str_list[3]<>'positive':
        print '4th column at row %d must be positive or negative' %i
      
        right=False
if right:
    print 'check passed...'
  
    
    
    
        
