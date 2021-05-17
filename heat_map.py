import numpy as np
import scipy.linalg as alg
import numpy.linalg as algo
import seaborn as sns
import matplotlib.pyplot as plt
from SBM import SBM
from Clustering import Methode_Spectrale

def erreur(p, q, n):
    model=SBM(p, q, n, 2)
    positifs, labels=Methode_Spectrale(model.M)
    #borne_sup=288*p/((p-q)**2)
    erreurs=0
    for result in positifs:
        if result > n/2:
            erreurs+=1
    return erreurs

def testeur(n , d):
    n= n + 1 if n%2==1 else n
    print(n)
    sample_p=np.linspace(0, 1, d)
    sample_q=np.linspace(0, 1, d)
    print(sample_q, sample_p)
    P, Q = np.meshgrid(sample_p, sample_q)
    Z=np.zeros([d,d])
    for i in range(d):
        for j in range(d):
            Z[i][j]=erreur(P[i][j],Q[i][j], n)
            #print(P[i][j],Q[i][j], Z[i][j])
    plt.xlabel("p")
    plt.ylabel("q")
    plt.pcolormesh(P, Q, Z)
    plt.colorbar()
    plt.title( "2-D Heat Map" )
    plt.show()

testeur(300, 41)


