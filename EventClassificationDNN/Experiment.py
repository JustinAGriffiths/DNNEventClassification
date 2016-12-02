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

#load in potential values not in default config
try : StopVal
except : StopVal=-1
try : NEpochsPerTrain
except : NEpochsPerTrain=Epochs
        
# Load the Data
from EventClassificationDNN.MultiClassTools import *

test_fraction=0.1
if not Train: test_fraction = 1
(Train_X, Train_Y), (Test_X, Test_Y), ClassIndex=LoadData(Samples,test_fraction,MaxEvents=MaxEvents,doshuffle=Train)

if not Train:
    Train_X=Test_X
    Train_Y=Test_Y
else :
    try :
        print 'Skimming :', len(Train_X), len(Train_Y)
        Train_X, Train_Y = CustomSkimFunction(Train_X, Train_Y)
        Test_X, Test_Y = CustomSkimFunction(Test_X, Test_Y)
        print 'Skimming :', len(Train_X), len(Train_Y)
    except :
        pass

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


# Normalize Ranges within variable groups e.g. masses, angles (phi, eta, cos separately)
# returned is the unnormalized data, Train_X/Test_X are modified
try : CustomNormFunc=CustomNormFunc
except : CustomNormFunc=None; pass

from NormalizeData import NormalizeData
try :
    normalization_dict = MyModel.MetaData["normalization_dict"]
except:
    normalization_dict = None
    pass
Train_X0, Test_X0, normalization_dict = NormalizeData(FieldGroups, Train_X, Test_X, normalization_dict=normalization_dict, NormMethod=NormMethod, NormFunc=CustomNormFunc)

print Train_X0['tau_0_pt'][0:3], Train_X['tau_0_pt'][0:3]

# Keep Only selected Variables
Train_X=Train_X[SelectedFields[VarSet]]
Test_X=Test_X[SelectedFields[VarSet]]

# Now Lets Simplify the structure (Note this requires everything to be a float)
# If you get an error that the input size isn't right, try changing float below to float32 or float64
Train_X=Train_X.view(np.float).reshape(Train_X.shape + (-1,))
Test_X=Test_X.view(np.float).reshape(Test_X.shape + (-1,))

# Protect against divide by zero! 
Train_X=np.nan_to_num(Train_X)
Test_X=np.nan_to_num(Test_X)

MyModel.MetaData["Config"]=Config
MyModel.MetaData["normalization_dict"]=normalization_dict

# Compile the Model
print "Compiling the Model... this will take a while."

optimizer="sgd"
MyModel.Compile(Loss=loss, Optimizer=optimizer)

model=MyModel.Model
# Print the summary
model.summary()

if Train:
    print "Training."
    from EventClassificationDNN.Analysis import AUC
    areas=[]
    import sys
    console=sys.stdout
    log=file('log','w')
    total_epochs=0
    if Epochs<0 :
        Epochs=1234567890
        print 'Setting Epochs to large number: ', Epochs
        pass
    if NEpochsPerTrain<0 : NEpochsPerTrain = Epochs
    epoch_bunches=[NEpochsPerTrain for i in xrange(0, Epochs/NEpochsPerTrain) ]
    epoch_bunches.append( Epochs%NEpochsPerTrain )
    for nepochs in epoch_bunches:
        sys.stdout=log
        total_epochs+=nepochs
        hist=MyModel.Train(Train_X, Train_Y, nepochs, BatchSize)
        sys.stdout=console
        areas.append(AUC(MyModel, Test_X, Test_Y, BatchSize))
        previous_auc=0
        if len(areas)>1 : previous_auc=areas[-2]
        auc = areas[-1]
        print 'Epocs: ', total_epochs, ' AUC: ', auc, ' delta: ', auc-previous_auc
        if (auc-previous_auc) < StopVal : break
        continue
    print areas
    #hist=MyModel.Train(Train_X, Train_Y, Epochs, BatchSize)
    
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
