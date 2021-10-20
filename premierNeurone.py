import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

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
#
#
#
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
#
#
#
"""
log_loss calcul le coût de notre model.
    param : - A (numpy.ndarray) : la matrice contenant les activation du model
            - y (numpy.ndarray) : la matrice contenant les bonnes réponses
    
    return: (numpy.float64) : le coût du model 
"""
def log_loss(A, y):
        return - 1 / len(y) * np.sum(y * np.log(A) - (1-y) * np.log(1-A))
#
#
#
"""
gradient calcule le gradient du poid W et du biai b,
utile pour réaliser le descente de gradiant nous permettant 
d'optimiser le poid et le biai
    param : - A (numpy.ndarray) : la matrice contenant les activation du model
            - X (numpy.ndarray) : la matrice contenant tout les sujet avec tout leur variable
            - y (numpy.ndarray) : la matrice contenant les bonnes réponses
    
    return: (dW, db) (tuple) -> (numpy.ndarray, numpy.float64) : ou dW est le gradiant de la matrice W et db le gradiant du biai
"""   
def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A-y)
    db = 1 / len(y) * np.sum(A-y)
    return (dW, db)
#    
#
#
"""
update met a jour les valeur de W et b avec le pas d'apprentissage
    param : - dW (numpy.ndarray) : la matrice contenant les gradiant de W
            - db (numpy.float64) : une flotant représentant le gradiant de b
            - W (numpy.ndarray) : la matrice contenant les poid de chaque variable
            - b (numpy.float64) : le biai
            - learning_rate (float) : le tout petit pas d'aprentissage
    return : (W, b) (tuple) -> () : lam matrice des poid et le biai mis a jour            
            
"""
def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return(W, b)
#
#
#
"""
predict permet d'avoir l'avis du model
    param : - X (numpy.ndarray) : la matrice contenant tout les sujet avec tout leur variable
            - W (numpy.ndarray) : la matrice contenant les poid de chaque variable
            - b : la biai
    return : (bool) -> true si catégorie 1
                    -> false si catégorie 0
"""
def predict (X, W, b):
    A = model(X, W, b)
    return A >= 0.5
#
#
#
"""
artificial_neuron est la fonction qui permet au model d'apprendre
    param : - X (numpy.ndarray) : la matrice contenant tout les sujet avec tout leur variable
            - y (numpy.ndarray) : la matrice contenant les bonnes réponses
            - learning_rate (float) : le pas d'apprentissage
            - n_iter (int) : le nombre de fois où l'on modifi le model
"""
def artificial_neuron(X, y, learning_rate = 0.1, n_iter  = 100):
    #init W, b
    W, b = initialisation(X)
    
    Loss = []
    
    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)
    
    y_pred = predict(X, W, b)
    print(f"La présition de ce model est de : {(accuracy_score(y, y_pred)) * 100} %")
    plt.plot(Loss)
    plt.show()
    
    x0 = np.linspace(-1, 4, 100)
    x1 = (-W[0]*x0 -b) / W[1]
    
    nP = np.array([2, 1])
    plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
    plt.scatter(nP[0], nP[1], c='r')
    plt.plot(x0, x1, c='orange', lw=3)
    plt.show()
    p = predict(nP, W, b)
    print(p)
    

    
    
#Main
artificial_neuron(X, y)
    