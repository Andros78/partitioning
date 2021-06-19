import numpy as np
import scipy.linalg as alg
import numpy.linalg as algo
import seaborn as sns
import matplotlib.pyplot as plt
from SBM import SBM
from Clustering import Methode_Spectrale


from sklearn.metrics import adjusted_rand_score
"""
    Details sur la fonction adjusted_rand_score:
    Entrée: true_labels: bonne partition et labels: partition de la méthode spectrale
    Sortie: score de la partition qui mesure l'erreur de la partition
    Le Score varie entre 0 et 1:
        -si Score = 0 alors la partition est aléatoire
        -si Score = 1 alors la partition est parfaite
    Pour plus d'information: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
"""
    

def heat_map_generator(n , resolution):
    """
    Génere une carte de chaleur: 
    -axe Ox correspond à un échantillon de valeurs pour p: probabilité des liens intra-communauté
    -axe Oy correspond à un échantillon de valeurs pour q: probabilité des liens inter-communauté
    -la couleur de la coordonnée (x,y) correspond au score de la partition par Méthode spéctrale
    sur un SBM de paramètres (p=x, q=y, n=n, k=2)

    NB1: Le score est mesurée par la métrique sklearn.metrics.adjusted_rand_score 
    NB2: L'échantillon de probabilité a pour taille la variable resolution
    """
    n= n + 1 if n%2==1 else n
    print("nombres d'individus:", n)
    sample_p=np.linspace(0, 1, resolution)
    sample_q=np.linspace(0, 1, resolution)

    P, Q = np.meshgrid(sample_p, sample_q)
    Z=np.zeros([resolution,resolution])
    true_labels=np.array([0]*int(n/2) + [1]*int(n/2))
    for i in range(resolution):
        for j in range(resolution):
            model=SBM(P[i][j],Q[i][j], n, 2)
            labels=Methode_Spectrale(model.M)
            Z[i][j]=adjusted_rand_score(true_labels, labels)
                 # = score de la Methode Spectrale sur un SBM de paramètre p=P[i][j] q=Q[i][j] 
    

    plt.xlabel("p: probabilité des liens intra-communauté")
    plt.ylabel("q: probabilité des liens inter-communauté")
    plt.pcolormesh(P, Q, Z)
    plt.colorbar()
    #barre_score.label("Score")
    plt.title( "Score de la Méthode Spectrale en fonction de p et q" )
    plt.show()

heat_map_generator(100, 101)


