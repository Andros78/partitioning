import numpy as np
import scipy.linalg as alg
import numpy.linalg as algo
import seaborn as sns
import matplotlib.pyplot as plt
from SBM import SBM
from Clustering import Methode_Spectrale
from sklearn.metrics import adjusted_rand_score

def erreur(p, q, n):
    model=SBM(p, q, n, 2)
    labels=Methode_Spectrale(model.M)
    labels_true=np.array([0]*int(n/2) + [1]*int(n/2))
    return adjusted_rand_score(labels_true, labels)
    #borne_sup=288*p/((p-q)**2)
    

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
    plt.xlabel("p: probabilité des liens intra-communauté")
    plt.ylabel("q: probabilité des liens inter-communauté")
    plt.pcolormesh(P, Q, Z)
    plt.colorbar()
    #barre_score.label("Score")
    plt.title( "Score de la Méthode Spectrale en fonction de p et q" )
    plt.show()

testeur(100, 101)


