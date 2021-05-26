import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans



def Methode_Spectrale(M):
    """ 
    Entrée: Matrice d'ajacence
    Sortie: Partition des individus en 2 communautés par la Méthode Spéctrale
    """

    n=len(M)
    #selection de la seconde plus grande valeur propre en valeur absolue 
    # de M et de son vecteur propre associé         
    VP2_pos, VectP2_pos=eigh(M, subset_by_index=[n-2,n-2])  
    VP2_neg, VectP2_neg=eigh(M, subset_by_index=[0,0])
    if abs(VP2_pos) > abs(VP2_neg):
        VP2_abs_max, VectP2_abs_max=abs(VP2_pos),  VectP2_pos
    else:
        VP2_abs_max, VectP2_abs_max=abs(VP2_neg),  VectP2_neg

    label=[0]*n
    for i in range(n):
        if VectP2_abs_max[i]<0:
            label[i]=1
    return label


def Methode_K_means(M, mtype="L"):
    """
    Entrée: Matrice Laplacienne ou Laplacienne Normalisée ou d'Adjacence
    Sortie: Partition des individus en 2 communautés (peut être généralisé à K)
    """
    n=len(M)
    if mtype=="L":
        start_index=1
        end_index=1
    if mtype=="A":
        start_index=n-2
        end_index=n-2
    
    vp, U=eigh(M, subset_by_index=[start_index, end_index])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(U)
    labels=kmeans.labels_
    return labels


#Sp2, Vect2=algo.eigh(self.mat)
#index_maxi=Sp2.index(max(Sp2))
#Sp2[index_maxi]=-10
#index_maxi2=Sp2.index(max(Sp2))
#print(Sp)
#print(Sp2[index_maxi2])

#Sp, Vect=alg.eigh(self.mat, subset_by_index=[1,1])
#print(Vect, Vect2)
