import numpy as np

class SBM:
    """ 
    Génère un graphe aléatoire type Stochastic Block Model. 

    Les paramètres en entrée sont les suivants:
    p: la probabilité des liens inter-communauté,
    q: la probabilité d'un lien intra-communauté
    n: le nombre de noeud
    k: le nombre de communautés
    """
    def __init__(self, p, q, n, k):
        self.k=k
        self.p=p
        self.q=q
        self.n= n + k - n%k if n%k!=0 else n    # ainsi n est divisible par k
        self.len_communities=int(self.n / self.k)

        self.Probability_Matrix()
        self.Adjacency_Matrix()
        self.Degree_Matrix()
        self.Laplacian_Matrix()
        self.Normalized_Laplacian()
    

    def Probability_Matrix(self):
        self.P=None  #Matrice de probabilité
        J=np.ones((self.len_communities, self.len_communities))
        tab=[None]*self.k
        for i in range(self.k):
            tab[i]=np.block([(self.p*J if i==j else self.q*J) for j in range(self.k)])
        self.P=np.block([[tab[i]] for i in range(self.k)]) - self.p*np.eye(self.n)
    
    def Adjacency_Matrix(self):
        self.M=np.zeros((self.n, self.n), dtype=np.int) #Matrice d'Adjacence
        L=np.random.random(size=(self.n, self.n))
        for i in range(self.n):
            for j in range(i+1):
                self.M[i][j]= 1 if L[i][j] <= self.P[i][j] else 0
                self.M[j][i]=self.M[i][j]

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
model=SBM(0.9, 0.1, 7, 3)
model.Probability_Matrix()
model.Adjacency_Matrix()
print(model.n)
print(model.A)
print(model.M)
"""