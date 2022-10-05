import  pandas as  pd
import numpy as np
from lce import LCEClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import normalize
import copy
def getlog(data):
    log=[]
    l=[]
    id=data[0][0]
    c=1
    for i in data:
        if id!=i[0]:
            id=i[0]
            log.append(l)
            l=[]
        if c==len(data):
            log.append(l)
        l.append(i[1])
        c+=1
    return log

def getallindex(l1,m):  #获得列表中所有元素的下标,l1是列表，m是要查的元素
    length=len(l1)
    n=l1.count(m)
    indexn=[]
    index=0
    for i in list(range(n)):
        index=l1.index(m,index,length)
        indexn.append(index)
        index+=1
    return indexn

def inittaglist(eventistlen,tracelen):             #初始化tag
    l=[]
    for i in range(tracelen):
        j=[]
        for i in range(eventistlen):
            j.append(0)
        l.append(j)
    return l

def gettraceevent(trace):
    l=[]
    for i in trace:
        if i not in l:
            l.append(i)
    return l

def getslsc(trace,eventlist):
    length=len(eventlist)
    tracelen=len(trace)
    sls = inittaglist(length,tracelen)
    slc=inittaglist(length,tracelen)
    el=gettraceevent(trace)
    for e in trace:
        l1=getallindex(trace,e)
        for j in el:
            if j!=e:
                l2=getallindex(trace,j)
                if max(l1)<min(l2) or min(l1)>max(l2):
                    sls[trace.index(e)][eventlist.index(j)]=1
                else:
                    slc[trace.index(e)][eventlist.index(j)] = 1
            else:
                if trace.count(e)!=1:
                    slc[trace.index(e)][eventlist.index(j)] = 1
    return sls,slc

def getadj(trace,eventlist):
    length=len(eventlist)
    tracelen=len(trace)
    adjlist=inittaglist(length,tracelen)
    for e in trace:
        l1 = getallindex(trace, e)
        for i in l1:
            if i!=tracelen-1:
                a=trace[i+1]
                adjlist[trace.index(e)][eventlist.index(a)] = 1
    return adjlist

def getfc(log,eventlist):
    codelist=[]
    for trace in log:
        l=[]
        sls,slc=getslsc(trace,eventlist)
        adj=getadj(trace,eventlist)
        for i in sls:
            for j in i:
                l.append(j)
        for i in slc:
            for j in i:
                l.append(j)
        for i in adj:
            for j in i:
                l.append(j)
        codelist.append(l)
    return codelist

def getaggf(data,ldata,m):
    fdata=[]
    label=[]
    l=[]
    c=0
    for i in data:
        c+=1
        for j in i:
            l.append(j)
        if c==m:
            c=0
            fdata.append(l)
            label.append(ldata[data.index(i)])
            l=[]
    return fdata,label

def gettrindata1(fc,data):
    print(len(fc),len(data))
    length=len(fc)
    trian=[]
    for i in range(length):
        l=[]
        for j in fc[i]:
            l.append(j)
        for j in data[i]:
            l.append(j)
        trian.append(l)
    return trian

def gettrindata0(data):
    length=len(data)
    trian=[]
    for i in range(length):
        l=[]
        for j in data[i]:
            l.append(j)
        trian.append(l)
    return trian
t1=time.time()

def getv(data):
    l=[]
    for i in data:
        if i not  in l:
            l.append(i)
    if len(l)==1:
        print('数据集有问题')
    else:
        print("ok")

filename=30
name='{}'.format(str(filename))
cat_cols =["Activity", "Resource"]
num_cols =["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]

data = pd.read_csv('{}.csv'.format(name), low_memory=False)
data = data.sort_values(by=['Case ID', 'event_nr'])
data.loc[data['label'] == 'regular', 'label'] = 0
data.loc[data['label'] == 'deviant', 'label'] = 1
ldata = list(data['label'].values)
eventlist1 = data['Activity'].values.tolist()
eventlist = list(set(eventlist1))
datax = data.loc[:, ['Case ID', 'Activity']].values
log = getlog(datax)
fc = getfc(log, eventlist)
dt_numeric = data.groupby('Case ID')[num_cols].aggregate([np.mean, np.max, np.min, np.sum]).reset_index()
dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns.values]
dt_numeric.rename(columns={'Case ID_': 'Case ID'}, inplace=True)
clo = list(dt_numeric.columns)[1:]
print(dt_numeric[clo])
dclo = dt_numeric[clo].values
dclo = normalize(dclo, axis=1)
dn = pd.DataFrame(dclo, columns=clo)
dt_transformed = pd.get_dummies(data[cat_cols])
dt_transformed0=copy.copy(dt_transformed)
    # dt_transformed = pd.concat([dt_transformed, dn, dt_numeric[['Case ID']]], axis=1)
dt_transformed = pd.concat([dt_transformed, dn], axis=1)
dt_transformed = dt_transformed.fillna(0)
    # print(dt_transformed.columns)
lidata = []
dt_transformed = dt_transformed.values
for i in dt_transformed:
    lidata.append(list(i))
lidata0 = []
dt_transformed0 = dt_transformed0.values
for i in dt_transformed0:
    lidata0.append(list(i))

fdata, label = getaggf(lidata, ldata, filename)
fdata0, _= getaggf(lidata0, ldata, filename)
rdata = gettrindata1(fc, fdata)
rdata0 = gettrindata1(fc, fdata0)

label = np.array(label)
rdata = np.array(rdata)
rdata0 = np.array(rdata0)

fdata=np.array(fdata)
X_train, X_test, y_train, y_test = train_test_split(rdata, label, test_size=0.2)
getv(y_train)
getv(y_test)
