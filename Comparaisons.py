from numpy.lib.type_check import nan_to_num
from Clustering import Methode_Spectrale, Methode_K_means
from sklearn.cluster import KMeans
from SBM import SBM
from scipy.linalg import eigh
#import matplotlib.pyplot as plt
from time import time 

n=100
p=0.7
q=0.5
def Clustering_kmean(labels):
    global n
    erreurs=0
    sum=0
    for i in range(int(n/2)):
        if labels[i]==1:
            sum+=1
    if sum>=int(n/4):
        cle=1
    else:
        cle=0
    for i in range(int(n/2)):
        if labels[i]!=cle:
            erreurs+=1
    cle=not cle
    for i in range(int(n/2), n):
        if labels[i]!=cle:
            erreurs+=1
    return erreurs


def comparaisons_algo():
    model=SBM(p, q, n, 2)
    M=model.M
    L=model.L
    Ln=model.Ln

    t_start=time()
    positifs, label=Methode_Spectrale(M)
    borne_sup=288*p/((p-q)**2)
    print("Borne sup:", borne_sup)
    erreurs_sp=0
    for result in positifs:
        if result > n/2:
            erreurs_sp+=1
    print("erreurs sectrales:", erreurs_sp, "in: ", time()-t_start)

    t_start=time()
    labels=Methode_K_means(L)
    erreurs_L=Clustering_kmean(labels)
    print("erreurs Kmeans L:", erreurs_L, "in: ", time()-t_start)

    t_start=time()
    labels=Methode_K_means(M, mtype="A")
    erreurs_A=Clustering_kmean(labels)
    print("erreurs adj:",erreurs_A, "in: ", time()-t_start)

    t_start=time()
    labels=Methode_K_means(Ln)
    erreurs_Ln=Clustering_kmean(labels)
    print("erreurs Ln", erreurs_Ln, "in: ", time()-t_start)

    return erreurs_sp, erreurs_A, erreurs_L, erreurs_Ln


N=10
erreurs_sp, erreurs_A, erreurs_L, erreurs_Ln=[0]*N, [0]*N, [0]*N, [0]*N
for i in range(N):
    erreurs_sp[i], erreurs_A[i], erreurs_L[i], erreurs_Ln[i]=comparaisons_algo()
print("erreurs_sp:",sum(erreurs_sp)/N)
print("erreurs_A:",sum(erreurs_A)/N)
print("erreurs_L:",sum(erreurs_L)/N)
print("erreurs_Ln:",sum(erreurs_Ln)/N)

"""
#plt.scatter(range(len(vp)), vp)
#plt.legend("Laplacian")

fig2=plt.subplot(312)
fig2.scatter(range(len(vpln)), vpln)
fig2.set_title("Laplacian Normalisé")

#plt.scatter(range(len(vp_adj)), vp_adj)
#plt.legend("Adjacence")

#plt.axis("equal")
#plt.show()

"""