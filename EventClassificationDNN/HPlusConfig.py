import random
import getopt
from DLTools.Permutator import *
import sys,argparse

#Need to define:
#Samples=[ [filename, treename[, classification_name ]], ...]
#FieldGroups=[ [ 'var1', 'var2', ], ...]
#SelectedFields=[ [ 'var1', 'var2', ], ...]
#Config={}
#Params={}

Samples=[
    ['/scratch/griffith/ntuples/341524/341524.root.h5', 'TrueTau', '341524'],
    ['/scratch/griffith/ntuples/410000/410000.root.h5', 'TrueTau', '410000'],
]

FieldGroups = [    
    ['tau_0_pt', 'met_et', 'ljet_0_pt',  'ljet_1_pt', 'bjet_0_pt'],
    ['tau_0_phi', 'met_phi', 'tau_0_eta', 'ljet_0_eta', 'ljet_0_phi', 'ljet_1_eta', 'ljet_1_phi', 'bjet_0_eta', 'bjet_0_phi'],
]

SelectedFields = [
    #populate the variables here that you want to train on
    ['tau_0_pt', 'met_et', 'tau_0_phi', 'met_phi'],    
]



Name="EventClassificationDNN"

Config={

    "MaxEvents":-1,
    "Epochs":4000, # -1 only quit when StopVal reached (not really implmented the -1 case, just setting to 1234567890)
    "NEpochsPerTrain":100, #Test for incremental improvements each NEpochs (-1 for one batch)
    "StopVal":1e-2, #quit training condition (if delta<StopVal, quit. set to -1 to ignore stopval)
    "BatchSize":2048*8,
    
    "LearningRate":0.005,
    
    "Decay":0.,
    "Momentum":0.,
    "Nesterov":0.,

    "WeightInitialization":"'normal'",
    "NormMethod":'"MinMax"',
}

Params={ "Width":[128],
         "Depth":[2],
         "loss":[#"'mean_squared_error'",  
                 '"categorical_crossentropy"'],
         }

def CustomNormFunc( FieldGroups, Train_X, Test_X, normalization_dict=None ) :
    print 'In custom norm function'

    GroupMins=[0]*len(FieldGroups)
    GroupMaxs=[0]*len(FieldGroups)

    if normalization_dict is None :
        normalization_dict = {}
    
        for Fs in xrange(0,len(FieldGroups)):
            Mins=[]
            Maxs=[]
            Means=[]
            Stds=[]
            for varI in xrange(0,len(FieldGroups[Fs])):
                Mins+=[np.min(Train_X[FieldGroups[Fs][varI]])]
                Maxs+=[np.max(Train_X[FieldGroups[Fs][varI]])]
                Means+=[np.average(Train_X[FieldGroups[Fs][varI]])]
                Stds+=[np.std(Train_X[FieldGroups[Fs][varI]])]
                continue
            for ival in xrange(0, len(Mins)):
                Mins[ival] = max(Mins[ival], Means[ival]-3*Stds[ival] )
                Maxs[ival] = min(Maxs[ival], Means[ival]+3*Stds[ival] )
                pass
            GroupMins[Fs]=min(Mins)
            GroupMaxs[Fs]=max(Maxs)

            
            normalization_dict[Fs] = [min(Mins), max(Maxs)]
            continue
        print 'normalization_dict: ', normalization_dict
        pass    
    else :
        print 'Loading previous normalization_dict'
        print 'normalization_dict: ', normalization_dict
        pass
    
    for Fs in xrange(0, len(FieldGroups)):
        for var in FieldGroups[Fs]:
            yy=Train_X[var]
            yy[:]= 1./(normalization_dict[Fs][1]-normalization_dict[Fs][0]) * (yy-normalization_dict[Fs][0])

            yy1=Test_X[var]
            yy1[:]= 1./(normalization_dict[Fs][1]-normalization_dict[Fs][0])* (yy1-normalization_dict[Fs][0])
            
            continue
        continue    
    
    return normalization_dict

def CustomSkimFunction(Train_X, Train_Y):

    skim_list_greater=[ ['tau_0_pt', 40000],
                        ['met_et', 100000],
                        ['n_jets', 2], ]

    row_keeps = np.array([True]*len(Train_X))
    for item in skim_list_greater :
        row_keeps*=(Train_X[item[0]] > item[1])
        
        continue    
    return Train_X[row_keeps], Train_Y[row_keeps]
    

PS=Permutator(Params)
Combos=PS.Permutations()

print "HyperParameter Scan: ", len(Combos), "possible combiniations."

if "HyperParamSet" in dir():
    i=int(HyperParamSet)
else:
    # Set Seed based on time
    random.seed()
    i=int(round(len(Combos)*random.random()))
    print "Randomly picking HyperParameter Set"
    

if i<0:
    print "SetList:"
    for j in xrange(0,len(Combos)):
        print j,":",Combos[j]

    quit()


print "Picked combination: ",i

for k in Combos[i]:
    Config[k]=Combos[i][k]

for MetaData in Params.keys():
    val=str(Config[MetaData]).replace('"',"")
    Name+="_"+val.replace("'","")

print "Model Filename: ",Name


