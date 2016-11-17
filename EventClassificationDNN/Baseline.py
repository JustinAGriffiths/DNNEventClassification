import h5py
import numpy as np

MaxEvents=10000

# Load the Data
from MultiClassTools import *

Files= [
    "/data/afarbin/JigSaw/h5/Diboson.h5",
    ("/data/afarbin/JigSaw/h5/GammaJet.h5", "GAMMA_SRAll"),
    "/data/afarbin/JigSaw/h5/QCD.h5",
    "/data/afarbin/JigSaw/h5/Top.h5",
    ("/data/afarbin/JigSaw/h5/Zjets.h5", "Z_SRAll"),
##    ("/data/afarbin/JigSaw/h5/Wjets.h5", "W_SRALL"),
    ("/data/afarbin/JigSaw/h5/SS_direct.h5","SS_direct_900_600_SRAll")
]

Samples=[]

for F in Files:
    if type(F) != str:
        name=F[1]
        F=F[0]
        Samples.append((F,name))
    else:
        name=F.split(".")[0].split("/")[-1]
        Samples.append((F,name+"_SRAll"))

Train_X=[]
Train_Y=[]
Test_X=[]
Test_Y=[]
X=[]

import pandas as pd


for i in xrange(0,len(Samples)):
    (aTrain_X, aTrain_Y), (aTest_X, aTest_Y), ClassIndex=LoadData([Samples[i]],0,MaxEvents=MaxEvents)

    # Train_X.append(aTrain_X)
    # Train_Y.append(aTrain_Y)
    # Test_X.append(aTest_X) 
    # Test_Y.append(aTest_Y) 
    X.append(pd.DataFrame(Train_X))

Fields=Train_X[0].dtype.names


cleaningCut="(abs(m_jet1_eta)>2.4 | m_jet1_chf/m_het1_FracSamplingMax>0.1)"
SRGluinoCommonCut="(NJet >= 4 & RPT_HT5PP<0.08 & HT5PP>800 & H2PP>550)"

Cuts= { "Cleaning":cleaningCut,
        "Gluino SR": SRGluinoCommonCut,
        "Cleaning + Gluino SR": cleaningCut+"&"+SRGluinoCommonCut }


aTable=[]
columns=["Cut", "Total","Passed","Efficiency"]


for x in X:
    for Cut in Cuts:
        Total=np.shape(x)[0]
        Passed=np.shape(x.query(Cuts[Cut]))[0]
    
        aTable.append([Cut,Total,Passed,Passed/Total])

import tabulate


tabulate.tabulate(aTable,headers= columns)
