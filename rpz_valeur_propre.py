import matplotlib.pyplot as plt
from SBM import SBM
import pylab
p, q, n, k=0.9, 0.1, 1000, 4
model=SBM(p, q, n, k)
from scipy.linalg import eigh
Spectre=eigh(model.M, eigvals_only=True)
x=range(model.n)
plt.hist(Spectre, bins=200)
plt.ylabel('Norme')
plt.yscale('log')
plt.title("Valeurs propres de M_adj à partir de SBM(p={} ; q={} ; n={} ; k={})".format(p,q,n,k))
plt.show()
