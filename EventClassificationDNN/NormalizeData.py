import numpy as np

def NormalizeData(FieldGroups, Train_X, Test_X, normalization_dict=None, NormMethod="MinMax", NormFunc=None):

    # Keep the original data before renomalizing... will use this in output
    Train_X0=Train_X.copy()
    Test_X0=Test_X.copy()

    if NormFunc is not None : return Train_X0, Test_X0, NormFunc(FieldGroups, Train_X, Test_X, normalization_dict)
    elif NormMethod=="MinMax" : return Train_X0, Test_X0, NormMinMax(FieldGroups, Train_X, Test_X, normalization_dict)
    

def NormMinMax( FieldGroups, Train_X, Test_X, normalization_dict=None ):
    GroupMins=[0]*len(FieldGroups)
    GroupMaxs=[0]*len(FieldGroups)

    if normalization_dict is None :
        normalization_dict = {}
    
        for Fs in xrange(0,len(FieldGroups)):
            Mins=[]
            Maxs=[]
            for varI in xrange(0,len(FieldGroups[Fs])):
                Mins+=[np.min(Train_X0[FieldGroups[Fs][varI]])]
                Maxs+=[np.max(Train_X0[FieldGroups[Fs][varI]])]
                continue
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

