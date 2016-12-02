import numpy as np

def CSVWriter(filename,X,Y,R):
    
    names=X.dtype.names

    colnames=""

    for n in names:
        colnames+=n+"/f:"

#    colnames+="TrueClass/f:"
    for i in xrange(0,Y.shape[1]):
        colnames+="true_"+str(i)+"/f:"

    for i in xrange(0,R.shape[1]):
        colnames+="predict_"+str(i)+"/f:"

    f = open(filename, 'w')
    f.write(colnames[:-1]+"\n")

    #X0=X.view(np.float32).reshape(X.shape + (-1,))
    X0=X.view(np.float).reshape(X.shape + (-1,))

    YI=np.nonzero(Y)[1]
    out=np.concatenate((X0,Y,R),axis=1)

    np.savetxt(f,out,delimiter=',')
