import numpy as np

class SBM:
    """where p,q probabilities of SBM, n total number of vertices
         and k total number of communities
    """
    def __init__(self, p, q, n, k):
        self.N_communities=k
        self.p=p
        self.q=q
        self.N_vertices= n + k - n%k if n%k!=0 else n #ainsi n est divisible par k
        #print("n:", self.N_vertices)
        self.len_communities=int(self.N_vertices / self.N_communities)
        self.A=None  #probability matrix
        self.M=np.zeros((self.N_vertices, self.N_vertices), dtype=np.int) #adjancency matrix  
        self.Probability_Matrix()
        self.Adjacency_Matrix()
        self.len=len(self.M)
        self.Degree_mat=None
        self.L=None
        self.Ln=None
        self.Laplacien_Matrix()
        
            
    def Degree_Matrix(self):
        self.Degree_mat=np.eye(self.len)
        for i in range(self.len):
            self.Degree_mat[i][i]=sum(self.M[i])

    def Laplacien_Matrix(self):
        self.Degree_Matrix()
        self.L=self.Degree_mat - self.M
        self.Normalized_Laplacian()

    def Normalized_Laplacian(self):
        Do=np.eye(self.len)
        for i in range(self.len):
            Do[i][i]=self.Degree_mat[i][i]**-1/2
        self.Ln = np.dot(np.dot(Do, self.L), Do)

    def Probability_Matrix(self):
        J=np.ones((self.len_communities, self.len_communities))
        tab=[None]*self.N_communities
        for i in range(self.N_communities):
            tab[i]=np.block([(self.p*J if i==j else self.q*J) for j in range(self.N_communities)])
        self.A=np.block([[tab[i]] for i in range(self.N_communities)]) - self.p*np.eye(self.N_vertices)
        
    def Adjacency_Matrix(self):
        L=np.random.random(size=(self.N_vertices, self.N_vertices))
        for i in range(self.N_vertices):
            for j in range(i+1):
                self.M[i][j]= 1 if L[i][j] <= self.A[i][j] else 0
                self.M[j][i]=self.M[i][j]

"""
model=SBM(0.9, 0.1, 7, 3)
model.Probability_Matrix()
model.Adjacency_Matrix()
print(model.N_vertices)
print(model.A)
print(model.M)
"""