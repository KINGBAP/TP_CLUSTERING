import numpy as np
import matplotlib . pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster

# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information

path = './clustering-benchmark/src/main/resources/datasets/artificial/'
databrut = arff.loadarff(open(path + "xclara.arff" , 'r'))
datanp = [[ x[0],x[1]] for x in databrut [0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]

f0 = [ x[0] for x in datanp ] # tous les elements de la premiere colonne
f1 = [ x[1] for x in datanp ] # tous les elements de la deuxieme colonne
plt.scatter( f0 , f1 , s = 8 )
plt.title(" Donnees initiales ")
plt.show()

#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
# f1 : valeur sur la deuxieme dimension
#

print ("Appel KMeans pour une valeur fixee de k")
tps1 = time.time()
k = 3
model = cluster.KMeans(n_clusters =k,init ='k-means++')
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_

plt.scatter (f0,f1,c = labels , s=8)
plt.title("Donnees apres clustering Kmeans")
plt.show()
print("nb clusters =" ,k , " , nb iter = " , iteration , " ,... ...runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ")