from numpy.lib.type_check import nan_to_num
from Algo_spectral import Algo_Spectral
from sklearn.cluster import KMeans
from SBM import SBM
from scipy.linalg import eigh
#import matplotlib.pyplot as plt
from time import time 

n=1000
p=0.7
q=0.5
def Clustering_kmean(U):
    global n
    kmeans = KMeans(n_clusters=2, random_state=0).fit(U)
    labels=kmeans.labels_
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
    algo=Algo_Spectral(M)
    positifs=algo.Partitionning()
    borne_sup=288*p/((p-q)**2)
    print("Borne sup:", borne_sup)
    erreurs_sp=0
    for result in positifs:
        if result > n/2:
            erreurs_sp+=1
    print("erreurs sectrales:", erreurs_sp, "in: ", time()-t_start)

    t_start=time()
    vp, U_L=eigh(L, subset_by_index=[0,2])
    erreurs_L=Clustering_kmean(U_L)
    print("erreurs Kmeans L:", erreurs_L, "in: ", time()-t_start)

    t_start=time()
    vp_adj, U_A=eigh(M, subset_by_index=[n-2, n-1])
    erreurs_A=Clustering_kmean(U_A)
    print("erreurs adj:",erreurs_A, "in: ", time()-t_start)

    t_start=time()
    vpln, U_Ln=eigh(Ln, subset_by_index=[0, 2])
    erreurs_Ln=Clustering_kmean(U_Ln)
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