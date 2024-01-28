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




def agglo_clustering_clusters(filename):
    datanp, f0, f1 = get_data(filename)

    clusters = range(2, 30)
    linkages = ['single', 'average', 'complete', 'ward']
    leaves = []

    best_silhouette = {linkage: None for linkage in linkages}
    best_k_silhouette = {linkage: None for linkage in linkages}
    best_labels_silhouette = {linkage: None for linkage in linkages}

    best_calinski = {linkage: None for linkage in linkages}
    best_k_calinski = {linkage: None for linkage in linkages}
    best_labels_calinski = {linkage: None for linkage in linkages}

    best_davies = {linkage: None for linkage in linkages}
    best_k_davies = {linkage: None for linkage in linkages}
    best_labels_davies= {linkage: None for linkage in linkages}

    total_time = {linkage: [] for linkage in linkages}
    silhouette_scores = {linkage: [] for linkage in linkages}
    calinski_scores = {linkage: [] for linkage in linkages}
    davies_scores = {linkage: [] for linkage in linkages}

    for k in clusters:
        for linkage in linkages:
            print(f"Appel AgglomerativeClustering pour n_clusters={k}, linkage={linkage}")
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(linkage=linkage, n_clusters=k)
            model = model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_
            leaves = model.n_leaves_

            total_time[linkage].append(round((tps2 - tps1) * 1000, 2))
            silhouette_scores[linkage].append(metrics.silhouette_score(datanp, labels, metric='euclidean'))
            calinski_scores[linkage].append(metrics.calinski_harabasz_score(datanp, labels))
            davies_scores[linkage].append(metrics.davies_bouldin_score(datanp, labels))

            # Update best scores
            if best_silhouette[linkage] is None or silhouette_scores[linkage][-1] > best_silhouette[linkage]:
                best_silhouette[linkage] = silhouette_scores[linkage][-1]
                best_k_silhouette[linkage] = k
                best_labels_silhouette[linkage] = labels

            if best_calinski[linkage] is None or calinski_scores[linkage][-1] > best_calinski[linkage]:
                best_calinski[linkage] = calinski_scores[linkage][-1]
                best_k_calinski[linkage] = k
                best_labels_calinski[linkage] = labels

            if best_davies[linkage] is None or davies_scores[linkage][-1] < best_davies[linkage]:
                best_davies[linkage] = davies_scores[linkage][-1]
                best_k_davies[linkage] = k
                best_labels_davies[linkage] = labels

    # Affichage clustering avec distance 
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    for linkage in linkages:
        axs[0].plot(clusters, silhouette_scores[linkage], label=linkage)
        axs[1].plot(clusters, calinski_scores[linkage], label=linkage)
        axs[2].plot(clusters, davies_scores[linkage], label=linkage)
        axs[3].plot(clusters, total_time[linkage], label=linkage)

    axs[0].set_title("Silhouette Score")
    axs[0].legend()
    axs[1].set_title("Calinski Harabasz Score")
    axs[1].legend()
    axs[2].set_title("Davies Bouldin Score")
    axs[2].legend()
    axs[3].set_title("Total Time (ms)")
    axs[3].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # Affichage du meilleur score et le nombre de clusters associé pour chaque métrique
    for linkage in linkages:
        print(f"Meilleur Silhouette Score pour {linkage}: {best_silhouette[linkage]} avec k={best_k_silhouette[linkage]} clusters")
        print(f"Meilleur Calinski Score pour {linkage}: {best_calinski[linkage]} avec k={best_k_calinski[linkage]} clusters")
        print(f"Meilleur Davies Score pour {linkage}: {best_davies[linkage]} avec k={best_k_davies[linkage]} clusters")

    print("nb clusters =", k, ", nb feuilles =", leaves, " runtime =", round((tps2 - tps1) * 1000, 2), " ms")

    fig, axs = plt.subplots(4, 3, figsize=(15, 15))

    for i, linkage in enumerate(linkages):
        axs[i, 0].scatter(f0, f1, c=best_labels_silhouette[linkage], s=8)
        axs[i ,0].set_title(f"Silhouette - {linkage} (k={best_k_silhouette[linkage]})")

        axs[i, 1].scatter(f0, f1, c=best_labels_calinski[linkage], s=8)
        axs[i, 1].set_title(f"Calinski - {linkage} (k={best_k_calinski[linkage]})")

        axs[i, 2].scatter(f0, f1, c=best_labels_davies[linkage], s=8)
        axs[i, 2].set_title(f"Davies - {linkage} (k={best_k_davies[linkage]})")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def agglo_clustering_distance(filename):
    datanp, f0, f1 = get_data(filename)
    min_dist, max_dist = dendrogramme(datanp)
    distance_sample = np.linspace(min_dist, max_dist, 30)


    # clustering hierarchique
    linkages = ['single', 'average', 'complete', 'ward']
    
    # Dictionnaires pour stocker les résultats spécifiques à chaque linkage
    total_time = {linkage: [] for linkage in linkages}
    silhouette_scores = {linkage: [] for linkage in linkages}
    calinski_scores = {linkage: [] for linkage in linkages}
    davies_scores = {linkage: [] for linkage in linkages}

    # Affichage clustering avec distance
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    # Inversion des boucles pour distance et linkage
    for linkage in linkages:
        # Liste pour stocker les distances filtrées spécifiques à chaque linkage
        distance_sample_filtered_linkage = []
        for distance in distance_sample:
            print(f"Appel AgglomerativeClustering pour distance={distance}, linkage={linkage}")
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(distance_threshold=distance, linkage=linkage, n_clusters=None)
            model = model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_
            k = model.n_clusters_

            if 1 < k <= 500:
                total_time[linkage].append(round((tps2 - tps1) * 1000, 2))
                silhouette_scores[linkage].append(metrics.silhouette_score(datanp, labels, metric='euclidean'))
                calinski_scores[linkage].append(metrics.calinski_harabasz_score(datanp, labels))
                davies_scores[linkage].append(metrics.davies_bouldin_score(datanp, labels))
                distance_sample_filtered_linkage.append(distance)
        
        axs[0].plot(distance_sample_filtered_linkage, silhouette_scores[linkage], label=linkage)
        axs[1].plot(distance_sample_filtered_linkage, calinski_scores[linkage], label=linkage)
        axs[2].plot(distance_sample_filtered_linkage, davies_scores[linkage], label=linkage)
        axs[3].plot(distance_sample_filtered_linkage, total_time[linkage], label=linkage)




    axs[0].set_title("Silhouette Score")
    axs[0].legend()
    axs[1].set_title("Calinski Harabasz Score")
    axs[1].legend()
    axs[2].set_title("Davies Bouldin Score")
    axs[2].legend()
    axs[3].set_title("Total Time (ms)")
    axs[3].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


    #     #plt.scatter ( f0 , f1 , c = labels , s = 8 )
    #     #plt.title(" Resultat du clustering ")
    #     #plt.show()
    #     #print (" nb clusters = " ,k ," , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


def dendrogramme(datanp) :

    import scipy.cluster.hierarchy as shc
    from scipy.cluster.hierarchy import fcluster
    # Donnees dans datanp
    print ( " Dendrogramme ’ single ’ donnees initiales " )
    linked_mat = shc.linkage (datanp ,'single')
    plt.figure (figsize=(12,12))
    dendrogram = shc.dendrogram (linked_mat,orientation = 'top' ,distance_sort='descending',show_leaf_counts =False)
    plt.show ()


    min_distance = min(linked_mat[:, 2])
    max_distance = max(linked_mat[:, 2])
    print("Distance minimale :", max_distance)

    return min_distance, max_distance
    


import os

path = './clustering-benchmark/src/main/resources/datasets/artificial/'
files1 = os.listdir(path)
first_part = [f for f in files1 if os.path.isfile(os.path.join(path, f))]

dataset_filenames = ["curves1.arff","circle.arff","elliptical_10_2.arff","R15.arff","cuboids.arff","dartboard1.arff","zelnik1.arff", "cluto-t5-8k.arff", "donut3.arff", "s-set1.arff","donutcurves.arff"]

path = './dataset-rapport/'
files2 = os.listdir(path)
second_part = [f for f in files2 if os.path.isfile(os.path.join(path, f))]

for f in dataset_filenames:
    all_clustering(f)
#for f in second_part:
#    all_clustering(f)
#all_clustering("xclara.arff")
#all_clustering("x1.txt")
#all_clustering("x2.doc")


#disk-4500n.arff ne marche pas avec Kmeans ni agglomeratif, essaie de diviser en clusters de même taille
#curves1.arff marche bien avec Kmeans (silhouette et Davies) et agglomeratif clusters
#elliptical_10_2.arff marche bien avec Kmeans (silhouette et Calinski) et agglomeratif clusters (average complete et ward marchent mieux que single)
#R15.arff marche pas trop mal avec Kmeans (15 ou 8 clusters)
#cuboids ne marche pas avec Kmeans mais avec agglomeratif clusters et single pas mal (2) sinon no
#2d-4c.arff marche bien avec Kmeans (Silhouette et Davies)  et agglomeratif car vraiment distincts
#simplex marche bien sauf avec agglomeratif clusters et single bizarre il trouve 8
#dartboard nul nul nul
#zelnik pas mal avvec kmeans (davies) pareil avec agglomeratif clusters (complete et ward avec davies)
#zelnik1 nul nul nul
#chainlink nul
#cure-t0-2000n-2D.arff nul à part avec agglomeratif clusters et single
#hepta marche bien
#cluto-t5-8k wow niquel avec kmeans et calinski et pareil calinski avec agglomeratif clusters
#donut3 bien avec single encore 
#donutcurves sépare bien les deux clusters mais pas interne à chqaque séparation
#curves2 nul
#zelnik5 nul
#s-set1 bien avec kmeans