import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#Creation de notre dataset 
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0]), 1) # y represente la catégorie à laquelle appartient la plante.


plt.scatter(X[:,0], X[:, 1], c=y, cmap="summer")
plt.show()


"""
initialisation sert à initialiser le biai b et le poid W
    param : - X (numpy.ndarray) : la matrice contenant tout les sujet avec tout leur variable
    
    return: - (W,b) (tuple) -> (numpy.ndarray, numpy.ndarray) : un tuple contenant 
"""
def initialisation(X):
    W = np.random.randn(X.shape[1], 1) # Ici tout les variable partage le poid, toute les variable x1 partageron le poid w1
    b = np.random.randn(1)
    return(W,b)

"""
model calcule le vector Z (le shema du neurone Z = X.W+b)
Puis calcul a parti de Z le vector d'activation A (pour savoir si le neurone d'active ou pas LOGISTIQUE)
    param : - X (numpy.ndarray) : la matrice contenant tout les sujet avec tout leur variable
            - W (numpy.ndarray) : la matrice contenant les poid de chaque variable
            - b (numpy.ndarray) : le biai
    
    return: A (numpy.ndarray) : La matrice representant la reponse du neuronne par catégorisé un sujet. (matrice de la fonciton d'activation)
                                                                                                        
"""
def model(X,W,b):
    Z = X.dot(W)+b
    A = 1 / (1 + np.exp(-Z))
    return A


W, b = initialisation(X)
A = model(X, W, b)
print(type(A))