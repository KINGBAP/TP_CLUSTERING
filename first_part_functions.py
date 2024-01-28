import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster, metrics


def get_data(filename):

    if filename.endswith(".arff"):
        print(f"Traitement du jeu de données {filename} arff")
        path = './clustering-benchmark/src/main/resources/datasets/artificial/'
        databrut = arff.loadarff(open(path + filename , 'r')) #xclara, impossible, triangle1 // pour target, spiral, cassini, xor, circle le Kmeans a du mal, ce qui fait qu'il ne converge pas vers un nombre de clusters le meilleur possible.
        datanp = [[ x[0],x[1]] for x in databrut [0]]

        f0 = [ x[0] for x in datanp ] # tous les elements de la premiere colonne
        f1 = [ x[1] for x in datanp ] # tous les elements de la deuxieme colonne
        plt.scatter( f0 , f1 , s = 8 )
        plt.title(" Donnees initiales ")
        plt.show()

    elif filename.endswith(".txt"):

        path = './dataset-rapport/'
        print(f"Traitement du jeu de données {filename} txt")
        databrut = np.loadtxt(path + filename, unpack=True)
        datanp = np.column_stack((databrut[0], databrut[1]))

        f0 = [x[0] for x in datanp]  # tous les elements de la premiere colonne
        f1 = [x[1] for x in datanp]  # tous les elements de la deuxieme colonne
        plt.scatter(f0, f1, s=8)
        plt.title(" Donnees initiales ")
        plt.show()
    else:
        print("Extension de fichier non prise en charge.")
        exit()

    return datanp, f0, f1

def all_clustering(filename) :

    kmeans_method(filename)
    agglo_clustering_clusters(filename)
    #agglo_clustering_distance(filename)


def kmeans_method(filename):

    datanp, f0, f1 = get_data(filename)

    ##### KMEANS method

    iteration = []
    total_time = []
    silhouette_score = []
    calinski_score = []
    davies_score = []
    K = range(2, 20)

    best_k_silhouette = None
    best_k_calinski = None
    best_k_davies = None
    best_labels_silhouette = None
    best_labels_calinski = None
    best_labels_davies = None
    best_silhouette = None
    best_calinski = None
    best_davies = None

    for k in K:
        print("Appel KMeans pour une valeur fixee de k")
        tps1 = time.time()
        model = cluster.KMeans(n_clusters=k, init='k-means++')
        model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        iteration.append(model.n_iter_)
        total_time.append(round((tps2 - tps1) * 1000, 2))

        silhouette = metrics.silhouette_score(datanp, labels, metric='euclidean')  # between -1 and 1
        calinski = metrics.calinski_harabasz_score(datanp, labels)  # higher is better
        davies = metrics.davies_bouldin_score(datanp, labels)  # lower is better

        silhouette_score.append(silhouette)
        calinski_score.append(calinski)
        davies_score.append(davies)

        if best_k_silhouette is None or silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k_silhouette = k
            best_labels_silhouette = labels

        if best_k_davies is None or davies < best_davies:
            best_davies = davies
            best_k_davies = k
            best_labels_davies = labels

        if best_k_calinski is None or calinski > best_calinski:
            best_calinski = calinski
            best_k_calinski = k
            best_labels_calinski = labels

    fig, axs = plt.subplots(5, 1, figsize=(8, 15))

    axs[0].plot(K, iteration, marker='o')
    axs[0].set_title("Number of Iterations")
    axs[0].set_xlabel("Nombre de clusters (k)")
    axs[1].plot(K, total_time, marker='o')
    axs[1].set_title("Runtime (ms)")
    axs[1].set_xlabel("Nombre de clusters (k)")
    axs[2].plot(K, silhouette_score, marker='o')
    axs[2].set_title("Silhouette Score")
    axs[2].set_xlabel("Nombre de clusters (k)")
    axs[3].plot(K, calinski_score, marker='o')
    axs[3].set_title("Calinski Harabasz Score")
    axs[3].set_xlabel("Nombre de clusters (k)")
    axs[4].plot(K, davies_score, marker='o')
    axs[4].set_title("Davies Bouldin Score")
    axs[4].set_xlabel("Nombre de clusters (k)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].scatter(f0, f1, c=best_labels_silhouette, s=8)
    axs[0].set_title(f"Clustering K-Means pour k={best_k_silhouette} (Silhouette)")
    axs[1].scatter(f0, f1, c=best_labels_calinski, s=8)
    axs[1].set_title(f"Clustering K-Means pour k={best_k_calinski} (Calinski)")
    axs[2].scatter(f0, f1, c=best_labels_davies, s=8)
    axs[2].set_title(f"Clustering K-Means pour k={best_k_davies} (Davies)")

    counts = {best_k_silhouette: 1, best_k_calinski: 1, best_k_davies: 1}
    best_k_overall = max(counts, key=counts.get)
    print("Le k qui revient le plus souvent :", best_k_overall)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()




def agglo_clustering_clusters(filename) :

    datanp, f0, f1 = get_data(filename)

    ######## CLUSTERING AGGLOMERATIF avec NOMRE DE CLUSTERS FIXE
    clusters = range(2,30)

    linkages = ['single', 'average', 'complete', 'ward']
    leaves=[]
    total_time= []
    silhouette_scores = {linkage: [] for linkage in linkages}
    calinski_scores = {linkage: [] for linkage in linkages}
    davies_scores = {linkage: [] for linkage in linkages}
    for k in clusters:
        for linkage in linkages:  
            print(f"Appel AgglomerativeClustering pour n_clusters={k}, linkage={linkage}")  
            tps1 = time.time ()
            model = cluster.AgglomerativeClustering(linkage=linkage,n_clusters=k)
            model = model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_
            kres = model.n_clusters_
            leaves = model.n_leaves_


            silhouette_scores[linkage].append(metrics.silhouette_score(datanp, labels, metric='euclidean'))
            calinski_scores[linkage].append(metrics.calinski_harabasz_score(datanp, labels))
            davies_scores[linkage].append(metrics.davies_bouldin_score(datanp, labels))

    # Affichage clustering avec distance 
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    for linkage in linkages:
        axs[0].plot(clusters, silhouette_scores[linkage], label=linkage)
        axs[1].plot(clusters, calinski_scores[linkage], label=linkage)
        axs[2].plot(clusters, davies_scores[linkage], label=linkage)

    axs[0].set_title("Silhouette Score")
    axs[0].legend()
    axs[1].set_title("Calinski Harabasz Score")
    axs[1].legend()
    axs[2].set_title("Davies Bouldin Score")
    axs[2].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # Affichage clustering avec k
    plt.scatter (f0,f1,c = labels , s = 8 )
    plt.title ( " Resultat du clustering " )
    plt.show ()
    print ("nb clusters =" ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


def agglo_clustering_distance(filename) :

    datanp, f0, f1 = get_data(filename)

    distance_sample = np.linspace(0, 0.1, 5)
    #clustering hierarchique
    linkages = ['single', 'average', 'complete', 'ward']
    leaves=[]
    total_time= []
    silhouette_scores = {linkage: [] for linkage in linkages}
    calinski_scores = {linkage: [] for linkage in linkages}
    davies_scores = {linkage: [] for linkage in linkages}

    for distance in distance_sample:
        for linkage in linkages:
            print(f"Appel AgglomerativeClustering pour distance={distance}, linkage={linkage}")
            tps1 = time.time ()
            model = cluster.AgglomerativeClustering(distance_threshold=distance,linkage=linkage,n_clusters = None)
            model = model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_
            k = model.n_clusters_
            leaves.append(model.n_leaves_)
            total_time.append(round(( tps2 - tps1 ) * 1000 , 2))
            print(k)
            if k <= 1:
                print(f"Arrêt de la boucle pour distance={distance}, linkage={linkage} car k <= 1")
                break  # Sortir de la boucle interne

            silhouette_scores[linkage].append(metrics.silhouette_score(datanp, labels, metric='euclidean'))
            calinski_scores[linkage].append(metrics.calinski_harabasz_score(datanp, labels))
            davies_scores[linkage].append(metrics.davies_bouldin_score(datanp, labels))


    # Affichage clustering avec distance 
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    for linkage in linkages:

        axs[0].plot(distance_sample, silhouette_scores[linkage], label=linkage)
        axs[1].plot(distance_sample, calinski_scores[linkage], label=linkage)
        axs[2].plot(distance_sample, davies_scores[linkage], label=linkage)

    axs[0].set_title("Silhouette Score")
    axs[0].legend()
    axs[1].set_title("Calinski Harabasz Score")
    axs[1].legend()
    axs[2].set_title("Davies Bouldin Score")
    axs[2].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


    #     #plt.scatter ( f0 , f1 , c = labels , s = 8 )
    #     #plt.title(" Resultat du clustering ")
    #     #plt.show()
    #     #print (" nb clusters = " ,k ," , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


def dendrogramme(datanp) :

    import scipy.cluster.hierarchy as shc
    # Donnees dans datanp
    print ( " Dendrogramme ’ single ’ donnees initiales " )
    linked_mat = shc.linkage (datanp ,'single')
    plt.figure (figsize=(12,12))
    shc.dendrogram (linked_mat,orientation = 'top' ,distance_sort='descending',show_leaf_counts =False)
    plt.show ()
    


import os

path = './clustering-benchmark/src/main/resources/datasets/artificial/'
files1 = os.listdir(path)
first_part = [f for f in files1 if os.path.isfile(os.path.join(path, f))]

path = './dataset-rapport/'
files2 = os.listdir(path)
second_part = [f for f in files2 if os.path.isfile(os.path.join(path, f))]

# for f in first_part:
#     all_clustering(f)
for f in second_part:
    all_clustering(f)
#all_clustering("xclara.arff")
#all_clustering("x1.txt")
#all_clustering("x2.doc")
