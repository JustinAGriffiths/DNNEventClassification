import random
import getopt
from DLTools.Permutator import *
import sys,argparse

#Previously in InputFiles.py
# Define the Input Files
InputFiles=["mP1000_mC150_mX100.h5",
            "mP1000_mC400_mX100.h5",
            "mP1000_mC600_mX100.h5",
            "mP1000_mC900_mX100.h5",
            "mP1000_mC200_mX100.h5",
            "mP1000_mC500_mX100.h5",
            "mP1000_mC700_mX100.h5",
            "mP1000_mC950_mX100.h5",
            "mP1000_mC300_mX100.h5",
            "mP1000_mC550_mX100.h5",
            "mP1000_mC800_mX100.h5"]
Files=[]

# Select datasets (formerly TTrees in original ROOT file)


for InputData in InputFiles:
    InputData="/scratch/data-backup/afarbin/crogan/h5/"+InputData
    Files+= [
        [InputData, "AA_Gen"],
        [InputData, "AB_Gen"],
        [InputData, "BA_Gen"],
        [InputData, "BB_Gen"]
    ]

Samples=[]

for F in Files:
    if type(F) != str:
        name=F[1]
        F=F[0]
        Samples.append([F,name])
    else:
        name=F.split(".")[0].split("/")[-1]
        Samples.append([F,name+"_SRAll"])


#Previously in InputVars.py

# Select Variables To use in training

# On the JigSaw Dataset Fields will be the following:
#

FieldGroups = [    
    ['mP', 'mC', 'mX', 'METx', 'METy', 'L1_pT', 'L1_M', 'L2_pT','L2_M', 'B1_pT','B1_M', 'B2_pT','B2_M', 'MPP_AA', 'Eb_a_AA', 'Eb_b_AA', 'El_a_AA', 'El_b_AA', 'MPP_BB', 'Eb_a_BB', 'Eb_b_BB', 'El_a_BB', 'El_b_BB','MPP_AB', 'Eb_a_AB', 'Eb_b_AB', 'El_a_AB', 'El_b_AB', 'MPP_BA', 'Eb_a_BA', 'Eb_b_BA', 'El_a_BA', 'El_b_BA'],
    ['L1_eta', 'L2_eta',  'B1_eta','B2_eta'],
    ['L1_phi', 'L2_phi','B1_phi', 'B2_phi'],
    ['cosPP_AA', 'cosPa_AA', 'cosPb_AA', 'cosCa_AA', 'cosCb_AA','cosPP_BB', 'cosPa_BB', 'cosPb_BB', 'cosCa_BB', 'cosCb_BB','cosPP_AB', 'cosPa_AB', 'cosPb_AB', 'cosCa_AB', 'cosCb_AB','cosPP_BA', 'cosPa_BA', 'cosPb_BA', 'cosCa_BA', 'cosCb_BA'],
    ['dphi_PP_Pa_AA', 'dphi_PP_Pb_AA', 'dphi_Pa_Ca_AA', 'dphi_Pb_Cb_AA', 'dphi_PP_Pa_BB', 'dphi_PP_Pb_BB', 'dphi_Pa_Ca_BB', 'dphi_Pb_Cb_BB', 'dphi_PP_Pa_AB', 'dphi_PP_Pb_AB', 'dphi_Pa_Ca_AB', 'dphi_Pb_Cb_AB','dphi_PP_Pa_BA', 'dphi_PP_Pb_BA', 'dphi_Pa_Ca_BA', 'dphi_Pb_Cb_BA'] ]

SelectedFields = [
    ['mP', 'mC', 'mX',  'L1_pT', 'L1_eta', 'L1_phi', 'L1_M', 'L2_pT', 'L2_eta', 'L2_phi', 'L2_M', 'B1_pT', 'B1_eta', 'B1_phi', 'B1_M', 'B2_pT', 'B2_eta', 'B2_phi', 'B2_M', 'MPP_AA', 'Eb_a_AA', 'Eb_b_AA', 'El_a_AA', 'El_b_AA', 'cosPP_AA', 'cosPa_AA', 'cosPb_AA', 'cosCa_AA', 'cosCb_AA', 'dphi_PP_Pa_AA', 'dphi_PP_Pb_AA', 'dphi_Pa_Ca_AA', 'dphi_Pb_Cb_AA', 'MPP_BB', 'Eb_a_BB', 'Eb_b_BB', 'El_a_BB', 'El_b_BB', 'cosPP_BB', 'cosPa_BB', 'cosPb_BB', 'cosCa_BB', 'cosCb_BB', 'dphi_PP_Pa_BB', 'dphi_PP_Pb_BB', 'dphi_Pa_Ca_BB', 'dphi_Pb_Cb_BB', 'MPP_AB', 'Eb_a_AB', 'Eb_b_AB', 'El_a_AB', 'El_b_AB', 'cosPP_AB', 'cosPa_AB', 'cosPb_AB', 'cosCa_AB', 'cosCb_AB', 'dphi_PP_Pa_AB', 'dphi_PP_Pb_AB', 'dphi_Pa_Ca_AB', 'dphi_Pb_Cb_AB', 'MPP_BA', 'Eb_a_BA', 'Eb_b_BA', 'El_a_BA', 'El_b_BA', 'cosPP_BA', 'cosPa_BA', 'cosPb_BA', 'cosCa_BA', 'cosCb_BA', 'dphi_PP_Pa_BA', 'dphi_PP_Pb_BA', 'dphi_Pa_Ca_BA', 'dphi_Pb_Cb_BA'],

    ['MPP_AA', 'Eb_a_AA', 'Eb_b_AA', 'El_a_AA', 'El_b_AA', 'cosPP_AA', 'cosPa_AA', 'cosPb_AA', 'cosCa_AA', 'cosCb_AA', 'dphi_PP_Pa_AA', 'dphi_PP_Pb_AA', 'dphi_Pa_Ca_AA', 'dphi_Pb_Cb_AA', 'MPP_BB', 'Eb_a_BB', 'Eb_b_BB', 'El_a_BB', 'El_b_BB', 'cosPP_BB', 'cosPa_BB', 'cosPb_BB', 'cosCa_BB', 'cosCb_BB', 'dphi_PP_Pa_BB', 'dphi_PP_Pb_BB', 'dphi_Pa_Ca_BB', 'dphi_Pb_Cb_BB', 'MPP_AB', 'Eb_a_AB', 'Eb_b_AB', 'El_a_AB', 'El_b_AB', 'cosPP_AB', 'cosPa_AB', 'cosPb_AB', 'cosCa_AB', 'cosCb_AB', 'dphi_PP_Pa_AB', 'dphi_PP_Pb_AB', 'dphi_Pa_Ca_AB', 'dphi_Pb_Cb_AB', 'MPP_BA', 'Eb_a_BA', 'Eb_b_BA', 'El_a_BA', 'El_b_BA', 'cosPP_BA', 'cosPa_BA', 'cosPb_BA', 'cosCa_BA', 'cosCb_BA', 'dphi_PP_Pa_BA', 'dphi_PP_Pb_BA', 'dphi_Pa_Ca_BA', 'dphi_Pb_Cb_BA'],

    ['METx', 'METy', 'L1_pT', 'L1_eta', 'L1_phi',  'L2_pT', 'L2_eta', 'L2_phi', 'B1_pT', 'B1_eta', 'B1_phi',  'B2_pT', 'B2_eta', 'B2_phi'],

    ['MPP_AA', 'Eb_a_AA', 'Eb_b_AA', 'El_a_AA', 'El_b_AA', 'cosPP_AA', 'cosPa_AA', 'cosPb_AA', 'cosCa_AA', 'cosCb_AA', 'dphi_PP_Pa_AA', 'dphi_PP_Pb_AA', 'dphi_Pa_Ca_AA', 'dphi_Pb_Cb_AA'],

    ['mP', 'mC', 'mX','METx', 'METy', 'L1_pT', 'L1_eta', 'L1_phi',  'L2_pT', 'L2_eta', 'L2_phi', 'B1_pT', 'B1_eta', 'B1_phi',  'B2_pT', 'B2_eta', 'B2_phi'],
]



Name="EventClassificationDNN"

Config={

    "MaxEvents":50000,
    "Epochs":1000,
    "BatchSize":2048*8,
    
    "LearningRate":0.005,
    
    "Decay":0.,
    "Momentum":0.,
    "Nesterov":0.,

    "WeightInitialization":"'normal'"
}

Params={ "Width":[128],
         "Depth":[2],
         "loss":[#"'mean_squared_error'",  
                 '"categorical_crossentropy"'],
         }

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


