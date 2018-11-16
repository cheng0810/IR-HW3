
# coding: utf-8


import os
import numpy as np
import math
import random
from tqdm import tqdm


doc_dir = 'Document/'
query_dir = 'Query/'

#training model by collection.txt
train_set = []
train_voc = []
with open('Collection.txt') as file:
    train_set += [line.rstrip() for line in file]   #put all the data to an array 

for voc in range(len(train_set)):
    train_voc.extend(train_set[voc].split())
    
train_voc = list(set(train_voc))


#target documents to test
test_set = []
test_voc = []
doc_list = os.listdir(doc_dir)
for doc_name in doc_list:
    with open(doc_dir + doc_name) as file:
        for line in range(3):  #we don't want first three line data
            file.readline()   
        doc = file.read().replace('-1','').replace('\n','')    #replace the element we don't need
        test_set += [doc]
        test_voc.extend(doc.split())    #all doc's voc
        
test_voc = list(set(test_voc))

#query file
query_set = []
query_voc = []
query_list = os.listdir(query_dir)
for query_name in query_list:
    with open(query_dir + query_name) as file:
        query = file.read().replace('-1','')
        query_set += [query]
        query_voc.extend(query.split())

#all vocabulary
voc_all = []
voc_all.extend(test_voc)
voc_all.extend(query_voc)
voc_all.extend(train_voc)
voc_all = list(set(voc_all))

print(len(voc_all))

#just for me to understand
# voc_all = list(map(int,voc_all))
# voc_all.sort()
# voc_all = list(map(str,voc_all))


Topic = 30
Word = len(voc_all) 
Document = len(train_set)

p_w_t = np.random.uniform(0, 10, size=(Word, Topic))
p_t_d = np.random.uniform(0, 10, size=(Topic, Document))

for t in range(Topic):
    p_w_t_column_sum = np.sum(p_w_t[:,t])
    for i in  range(Word):
        p_w_t[i,t] /= p_w_t_column_sum
print(np.sum(p_w_t[:, 1]))

for t in range(Document):
    p_t_d_column_sum = np.sum(p_t_d[:,t])
    for i in  range(Topic):
        p_t_d[i,t] /= p_t_d_column_sum
print(np.sum(p_t_d[:, 1]))

p = np.zeros([Topic,Word,Document])
den = np.zeros([Word,Document])
def FI_Estep():
    for i in tqdm(range(Word)):
        for j in range(Document):
            Denominator = 0
            for k in range(Topic):
                p[k,i,j] = p_w_t[i,k]*p_t_d[k,j]
                Denominator += p[k,i,j]
                
            den[i,j] = Denominator
            for k in range(Topic):
                if Denominator !=0:
                    p[k,i,j] /= Denominator
                else:
                    p[k,i,j] = 0


c=np.zeros([Word,Document])
d = []
def FI_count():
    for j in tqdm(range(len(train_set))):
        file = train_set[j].split()
        d.append(len(file))
        for i in range(Word):
            c[i,j] = file.count(voc_all[i])

def FI_Mstep():

#處理p_w_t
    for k in tqdm(range(Topic)):
        temp1 = 0
        for i in range(Word):
            temp = 0
            for j in range(Document):
                temp += c[i,j]*p[k,i,j]  #分子sum
            p_w_t[i,k] = temp   #還沒處理分母 要把所有temp 加總
            temp1 += temp
        for i in range(Word):
            if temp1 != 0:
                p_w_t[i,k] /= temp1 
            else:
                p_w_t[i,k] = 0
            
#處理p_t_d
    for k in tqdm(range(Topic)):
        for j in range(Document):
            temp = 0
            temp1 = 0
            for i in range(Word):
                temp += c[i,j]*p[k,i,j] #存分子
                temp1 += c[i,j]
            if temp1 != 0:
                p_t_d[k,j] = temp/temp1
            else:
                p_t_d[k,j] = 0



FI_count()

for i in range(50):
    FI_Estep()
    FI_Mstep()
    
    
    
#finish training


test_document = len(test_set)

new_t_d = np.random.uniform(0, 10, size=(Topic, test_document))

for t in range(test_document):
    new_t_d_column_sum = np.sum(new_t_d[:,t])
    for i in  range(Topic):
        new_t_d[i,t] /= new_t_d_column_sum
print(np.sum(new_t_d[:, 1]))


new_p = np.zeros([Topic,Word,test_document])
new_den = np.zeros([Word,test_document])
def Estep():
    for i in tqdm(range(Word)):
        for j in range(test_document):
            Denominator = 0
            for k in range(Topic):
                new_p[k,i,j] = p_w_t[i,k]*new_t_d[k,j]
                Denominator += new_p[k,i,j]
                
            new_den[i,j] = Denominator
            for k in range(Topic):
                if Denominator !=0:
                    new_p[k,i,j] /= Denominator
                else:
                    new_p[k,i,j] = 0

new_c=np.zeros([Word,test_document])
new_d = []
def count():
    for j in tqdm(range(len(test_set))):
        file = test_set[j].split()
        new_d.append(len(file))
        for i in range(Word):
            c[i,j] = file.count(voc_all[i])

def Mstep():           
#處理p_t_d
    for k in tqdm(range(Topic)):
        for j in range(test_document):
            temp = 0
            temp1 = 0
            for i in range(Word):
                temp += new_c[i,j]*new_p[k,i,j] #存分子
                temp1 += new_c[i,j]
            if temp1 != 0:
                new_t_d[k,j] = temp/temp1
            else:
                new_t_d[k,j] = 0

count()

for i in range(10):
    Estep()
    Mstep()
    
    
#finish testing

p_q_d = np.zeros([16,test_document],dtype= object)
p_w_d = np.zeros([Word,test_document])

for i in range(Word):   #處理p_w_d
    for j in range(test_document):
        p_w_d[i,j] = new_c[i,j]/new_d[j]

#BGLM
file_BGLM = open('BGLM.txt')
BGLM = np.zeros([51253,2])
for i in range(51253):
    BGLM[i] = file_BGLM.readline().split()

alpha = 0.1
beta = 0.8
for i in tqdm(range(len(query_set))):
    file1 = query_set[i].split()
        
    for j in range(test_document):  #d=0~2264
        for q in range(len(file1)): #query1 file長度      file1(q)表query1內的文字
            for s in range(Word):
                if file1[q] == voc_all[s]:
                    p_q_d[i,j] += math.log(alpha*p_w_d[s,j]
                                           + beta*new_den[s,j]
                                           + (1-alpha-beta)*math.exp(BGLM[int(file1[q]),1]))       
                    break
        
p_q_d = pd.DataFrame(p_q_d,columns = doc_list , index = query_list)

p_q_d

f = open('M10715090.txt','w')    #write in txt file
f.write('Query,RetrievedDocuments\n')
for i in range(16):
    f.write(p_q_d.index[i])
    f.write(',')
    p_q_d = p_q_d.sort_values(by = query_list[i],ascending= False,axis = 1)
    for j in range(test_document):
        f.write(p_q_d.columns[j])
        f.write(' ')
    f.write('\n')

