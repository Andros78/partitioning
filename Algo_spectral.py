import numpy as np
import scipy.linalg as alg
import numpy.linalg as algo
import seaborn as sns
import matplotlib.pyplot as plt

class Algo_Spectral:
    def __init__(self, Adjancy_Matrix):
        self.mat=Adjancy_Matrix
        self.len=len(self.mat)
        self.Degree_mat=None
        self.Laplacian_mat=None
            
    def Degree_Matrix(self):
        self.Degree_mat=np.eye(self.len)
        for i in range(self.len):
            self.Degree_mat[i][i]=sum(self.mat[i])

    def Laplacien_Matrix(self):
        self.Degree_Matrix()
        self.Laplacian_mat=self.Degree_mat - self.mat

    def Partitionning(self):  #Methode spectrale
        n=self.len
        Sp, Vect=alg.eigh(self.mat, subset_by_index=[self.len-2, self.len-2])  #diagonalisation
        #Sp2, Vect2=algo.eigh(self.mat)
        #index_maxi=Sp2.index(max(Sp2))
        #Sp2[index_maxi]=-10
        #index_maxi2=Sp2.index(max(Sp2))
        #print(Sp)
        #print(Sp2[index_maxi2])

        #Sp, Vect=alg.eigh(self.mat, subset_by_index=[1,1])
        #print(Vect, Vect2)

        positifs=[]
        for i in range(self.len):
            if Vect[i]<0:
                positifs.append(i)
        return positifs