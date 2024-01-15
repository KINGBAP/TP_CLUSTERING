import numpy as np
import matplotlib . pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster, metrics

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
databrut = arff.loadarff(open(path + "xor.arff" , 'r')) #xclara, impossible // pour xor, le Kmeans a du mal, ce qui fait qu'il ne converge pas vers un nombre de clusters le meilleur possible.
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

iteration=[]
total_time= []
silhouette_score =[]
calinski_score =[]
davies_score =[]
K = range(2,10)

fig, axs = plt.subplots(5, 1, figsize=(8, 15))

for k in K :
    print ("Appel KMeans pour une valeur fixee de k")
    tps1 = time.time()
    model = cluster.KMeans(n_clusters =k,init ='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    iteration.append(model.n_iter_)
    total_time.append(round (( tps2 - tps1 ) * 1000 , 2 ))

    #plt.scatter(f0,f1,c=labels,s=8)
    #plt.title("Donnees apres clustering Kmeans")
    #plt.show()
    print("nb clusters =" ,k , " , nb iter = " , iteration , " ,... ...runtime = " , total_time ," ms ")

    silhouette_score.append(metrics.silhouette_score(datanp, labels, metric='euclidean')) #between -1 and 1
    calinski_score.append(metrics.calinski_harabasz_score(datanp, labels)) #higher is better
    davies_score.append(metrics.davies_bouldin_score(datanp, labels)) #lower is better 

axs[0].plot(K, iteration)
axs[0].set_title("Number of Iterations")
axs[1].plot(K, total_time)
axs[1].set_title("Runtime (ms)")
axs[2].plot(K, silhouette_score)
axs[2].set_title("Silhouette Score")
axs[3].plot(K, calinski_score)
axs[3].set_title("Calinski Harabasz Score")
axs[4].plot(K, davies_score)
axs[4].set_title("Davies Bouldin Score")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

