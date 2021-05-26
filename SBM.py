import numpy as np

class SBM:
    """ 
    Génère un graphe aléatoire type Stochastic Block Model. 

    Les paramètres en entrée sont les suivants:
    p: la probabilité des liens intra-communauté,
    q: la probabilité d'un lien inter-communauté
    n: le nombre de noeud
    k: le nombre de communautés
    """
    def __init__(self, p, q, n, k):
        self.k=k
        self.p=p
        self.q=q
        self.n= n + k - n%k if n%k!=0 else n    # ainsi n est divisible par k
        self.len_communities=int(self.n / self.k)
        
        self.Adjacency_Matrix()
        self.Degree_Matrix()
        self.Laplacian_Matrix()
        self.Normalized_Laplacian()
    
    def Adjacency_Matrix(self):
        self.M=np.zeros((self.n, self.n), dtype=np.int) #Matrice d'Adjacence
        L=np.random.random(size=(self.n, self.n))
        for i in range(self.n):
            for j in range(i+1):
                if i==j: #graphe sans boucle
                    self.M[i][j]=0
                elif i//self.len_communities == j//self.len_communities: # si même quotient par len_communities alors i et j sont de la même communauté
                    self.M[i][j]= 1 if L[i][j] <= self.p else 0
                else:
                    self.M[i][j]= 1 if L[i][j] <= self.q else 0
                self.M[j][i]=self.M[i][j] #symétrie de la matrice d'ajacence

    def Degree_Matrix(self):
        self.D=np.eye(self.n) #Matrice des degrés
        for i in range(self.n):
            self.D[i][i]=sum(self.M[i])

    def Laplacian_Matrix(self):
        self.L=self.D - self.M

    def Normalized_Laplacian(self):
        Do=np.eye(self.n)
        for i in range(self.n):
            Do[i][i]=self.D[i][i]**-1/2
        self.Ln = np.dot(np.dot(Do, self.L), Do)



"""
#Tester le SBM: 
model=SBM(0.8, 0.4, 10, 3)

print("nombre de noeuds:", model.n)
print("Matrice d'Ajacence:", model.M)
"""



