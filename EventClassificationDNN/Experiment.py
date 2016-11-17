import sys,os,argparse

import h5py
import numpy as np

# Parse Arguments
execfile("EventClassificationDNN/Arguments.py")

# Now load the Hyperparameters
execfile(ConfigFile)

if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Load the Data
from EventClassificationDNN.MultiClassTools import *

(Train_X, Train_Y), (Test_X, Test_Y), ClassIndex=LoadData(Samples,.1,MaxEvents=MaxEvents)

# Keep the original data before renomalizing... will use this in output
Train_X0=Train_X.copy()
Test_X0=Test_X.copy()

GroupMins=[0]*len(FieldGroups)
GroupMaxs=[0]*len(FieldGroups)

# Normalize Ranges within variable groups e.g. masses, angles (phi, eta, cos separately)
for Fs in xrange(0,len(FieldGroups)):
    Mins=[]
    Maxs=[]
    for varI in xrange(0,len(FieldGroups[Fs])):
        Mins+=[np.min(Train_X0[FieldGroups[Fs][varI]])]
        Maxs+=[np.max(Train_X0[FieldGroups[Fs][varI]])]

    GroupMins[Fs]=min(Mins)
    GroupMaxs[Fs]=max(Maxs)

    for var in FieldGroups[Fs]:
        yy=Train_X[var]
        yy[:]= 1./(GroupMaxs[Fs]-GroupMins[Fs]) * (yy-GroupMins[Fs])

        yy1=Test_X[var]
        yy1[:]= 1./(GroupMaxs[Fs]-GroupMins[Fs])* (yy1-GroupMins[Fs])

Train_X_N=Train_X

# Keep Only selected Variables
Train_X=Train_X[SelectedFields[VarSet]]
Test_X=Test_X[SelectedFields[VarSet]]

Train_X_S=Train_X

# Now Lets Simplify the structure (Note this requires everything to be a float)
# If you get an error that the input size isn't right, try changing float below to float32 or float64
Train_X=Train_X.view(np.float).reshape(Train_X.shape + (-1,))
Test_X=Test_X.view(np.float).reshape(Test_X.shape + (-1,))

# Protect against divide by zero! 
Train_X=np.nan_to_num(Train_X)
Test_X=np.nan_to_num(Test_X)

# Get some Inof
N_Inputs=len(SelectedFields[VarSet])
N_Classes=np.shape(Train_Y)[1]
print "N Inputs:",N_Inputs
print "N Classes:",N_Classes

# Now Build the Model
from DLTools.ModelWrapper import *

# Build the Model
from EventClassificationDNN.Classification import FullyConnectedClassification

if LoadModel:
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/":
        LoadModel=LoadModel[:-1]
    Name=os.path.basename(LoadModel)
    MyModel=ModelWrapper(Name)
    MyModel.InDir=LoadModel
    MyModel.Load()
else:
    MyModel=FullyConnectedClassification(Name,N_Inputs,Width,Depth,N_Classes,WeightInitialization)
    MyModel.Build()

MyModel.MetaData["Config"]=Config

# Compile the Model
print "Compiling the Model... this will take a while."

optimizer="sgd"
MyModel.Compile(Loss=loss, Optimizer=optimizer)

model=MyModel.Model
# Print the summary
model.summary()

if Train:
    print "Training."
    hist=MyModel.Train(Train_X, Train_Y, Epochs, BatchSize)
    
    score = model.evaluate(Test_X, Test_Y , batch_size=BatchSize)

    print "Final Score:",score
    
    MyModel.MetaData["FinalScore"]=score

# Save 
MyModel.Save()

# Analysis
from EventClassificationDNN.Analysis import MultiClassificationAnalysis
result=MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize )

# Dump out the predictions added to the input
if WriteResults:
    print "Writing Results."
    from EventClassificationDNN.CSVWriter import *
    CSVWriter(MyModel.OutDir+"/Result.csv",Test_X0,Test_Y,result)
