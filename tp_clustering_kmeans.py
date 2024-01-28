import numpy as np
import matplotlib.pyplot as plt
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
dataset_filenames = ["cassini.arff","aggregation.arff","circle.arff","fourty.arff","target.arff","spiral.arff","triangle1.arff", "impossible.arff", "xclara.arff", "tetra.arff"]

for dataset_filename in dataset_filenames:
    print(f"Traitement du jeu de données {dataset_filename}")

    databrut = arff.loadarff(open(path + dataset_filename , 'r')) #xclara, impossible, triangle1 // pour target, spiral, cassini, xor, circle le Kmeans a du mal, ce qui fait qu'il ne converge pas vers un nombre de clusters le meilleur possible.
    print(databrut)
    datanp = [[ x[0],x[1]] for x in databrut [0]]
    print(datanp)

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

    # ##### KMEANS method 

    # iteration=[]
    # total_time= []
    # silhouette_score =[]
    # calinski_score =[]
    # davies_score =[]
    # K = range(2,10)

    # best_k_silhouette = None
    # best_k_calinski = None
    # best_k_davies = None
    # best_labels_silhouette = None
    # best_labels_calinski = None
    # best_labels_davies = None
    # best_silhouette = None
    # best_calinski = None
    # best_davies = None

    # for k in K :
    #     print ("Appel KMeans pour une valeur fixee de k")
    #     tps1 = time.time()
    #     model = cluster.KMeans(n_clusters =k,init ='k-means++')
    #     model.fit(datanp)
    #     tps2 = time.time()
    #     labels = model.labels_
    #     iteration.append(model.n_iter_)
    #     total_time.append(round (( tps2 - tps1 ) * 1000 , 2 ))

    #     # plt.scatter(f0,f1,c=labels,s=8)
    #     # plt.title("Donnees apres clustering Kmeans")
    #     # plt.show()
    #     # print("nb clusters =" ,k , " , nb iter = " , iteration , " ,... ...runtime = " , total_time ," ms ")

    #     silhouette = metrics.silhouette_score(datanp, labels, metric='euclidean') #between -1 and 1
    #     calinski = metrics.calinski_harabasz_score(datanp, labels) #higher is better
    #     davies = metrics.davies_bouldin_score(datanp, labels) #lower is better 

    #     silhouette_score.append(silhouette)
    #     calinski_score.append(calinski)
    #     davies_score.append(davies)

    #     if best_k_silhouette is None or silhouette > best_silhouette:
    #         best_silhouette = silhouette
    #         best_k_silhouette = k
    #         best_labels_silhouette = labels

    #     if best_k_davies is None or davies < best_davies:
    #         best_davies = davies
    #         best_k_davies = k
    #         best_labels_davies = labels
        
    #     if best_k_calinski is None or calinski > best_calinski:
    #         best_calinski = calinski
    #         best_k_calinski = k
    #         best_labels_calinski = labels


    # fig, axs = plt.subplots(5, 1, figsize=(8, 15))

    # axs[0].plot(K, iteration, marker='o')
    # axs[0].set_title("Number of Iterations")
    # axs[0].set_xlabel("Nombre de clusters (k)")
    # axs[1].plot(K, total_time, marker='o')
    # axs[1].set_title("Runtime (ms)")
    # axs[1].set_xlabel("Nombre de clusters (k)")
    # axs[2].plot(K, silhouette_score, marker='o')
    # axs[2].set_title("Silhouette Score")
    # axs[2].set_xlabel("Nombre de clusters (k)")
    # axs[3].plot(K, calinski_score, marker='o')
    # axs[3].set_title("Calinski Harabasz Score")
    # axs[3].set_xlabel("Nombre de clusters (k)")
    # axs[4].plot(K, davies_score, marker='o')
    # axs[4].set_title("Davies Bouldin Score")
    # axs[4].set_xlabel("Nombre de clusters (k)")
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    # plt.show()


    # fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # axs[0].scatter(f0, f1, c=best_labels_silhouette, s=8)
    # axs[0].set_title(f"Clustering K-Means pour k={best_k_silhouette} (Silhouette)")
    # axs[1].scatter(f0, f1, c=best_labels_calinski, s=8)
    # axs[1].set_title(f"Clustering K-Means pour k={best_k_calinski} (Calinski)")
    # axs[2].scatter(f0, f1, c=best_labels_davies, s=8)
    # axs[2].set_title(f"Clustering K-Means pour k={best_k_davies} (Davies)")

    # counts = {best_k_silhouette: 1, best_k_calinski: 1, best_k_davies: 1}
    # best_k_overall = max(counts, key=counts.get)
    # print("Le k qui revient le plus souvent :", best_k_overall)

    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    # plt.show()


    ##### Clustering AGGLOMERATIF

    #Dendrogramme
    import scipy.cluster.hierarchy as shc
    # Donnees dans datanp
    print ( " Dendrogramme ’ single ’ donnees initiales " )
    linked_mat = shc.linkage (datanp ,'single')
    plt.figure (figsize=(12,12))
    shc.dendrogram (linked_mat,orientation = 'top' ,distance_sort='descending',show_leaf_counts =False)
    plt.show ()

    # Nombre de labels souhaité
    min_desired_labels = 2
    max_desired_labels = 100
    num_steps = 5

    # Utiliser shc.cut_tree pour extraire les distances associées à un certain nombre de leaders (labels)
    cut_tree_result = shc.cut_tree(linked_mat, n_clusters=[min_desired_labels, max_desired_labels])
    leaders_indices = np.unique(cut_tree_result)

    # Extraire les distances associées aux indices des leaders
    distance_range = linked_mat[leaders_indices, 2]

    # Générer un échantillon de valeurs dans la plage de distances souhaitée
    distance_sample = np.linspace(distance_range.min(), distance_range.max(), num_steps)
    print(distance_sample)

    # #clustering hierarchique
    # linkages = ['single', 'average', 'complete', 'ward']
    # leaves=[]
    # total_time= []
    # silhouette_scores = {linkage: [] for linkage in ['single', 'average', 'complete', 'ward']}
    # calinski_scores = {linkage: [] for linkage in ['single', 'average', 'complete', 'ward']}
    # davies_scores = {linkage: [] for linkage in ['single', 'average', 'complete', 'ward']}

    # for distance in distance_sample:
    #     for linkage in linkages:
    #         print(f"Appel AgglomerativeClustering pour distance={distance}, linkage={linkage}")
    #         tps1 = time.time ()
    #         model = cluster.AgglomerativeClustering(distance_threshold=distance,linkage=linkage,n_clusters = None)
    #         model = model.fit(datanp)
    #         tps2 = time.time()
    #         labels = model.labels_
    #         k = model.n_clusters_
    #         leaves.append(model.n_leaves_)
    #         total_time.append(round(( tps2 - tps1 ) * 1000 , 2))
    #         print(k)
    #         if k <= 1:
    #             print(f"Arrêt de la boucle pour distance={distance}, linkage={linkage} car k <= 1")
    #             break  # Sortir de la boucle interne

    #         silhouette_scores[linkage].append(metrics.silhouette_score(datanp, labels, metric='euclidean'))
    #         calinski_scores[linkage].append(metrics.calinski_harabasz_score(datanp, labels))
    #         davies_scores[linkage].append(metrics.davies_bouldin_score(datanp, labels))


    # # Affichage clustering avec distance 
    # fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # for linkage in ['single', 'average', 'complete', 'ward']:
    #     print(f"Longueur de distance_sample pour {linkage}: {len(distance_sample)}")
    #     print(f"Longueur de silhouette_scores[{linkage}]: {len(silhouette_scores[linkage])}")

    #     axs[0].plot(distance_sample, silhouette_scores[linkage], label=linkage)
    #     axs[1].plot(distance_sample, calinski_scores[linkage], label=linkage)
    #     axs[2].plot(distance_sample, davies_scores[linkage], label=linkage)

    # axs[0].set_title("Silhouette Score")
    # axs[0].legend()
    # axs[1].set_title("Calinski Harabasz Score")
    # axs[1].legend()
    # axs[2].set_title("Davies Bouldin Score")
    # axs[2].legend()
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    # plt.show()
    # #     #plt.scatter ( f0 , f1 , c = labels , s = 8 )
    # #     #plt.title(" Resultat du clustering ")
    # #     #plt.show()
    # #     #print (" nb clusters = " ,k ," , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


    ######## CLUSTERING AGGLOMERATIF avec NOMRE DE CLUSTERS FIXE
    clusters = range(2,10)

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
    plt.scatter ( f0 , f1 , c = labels , s = 8 )
    plt.title ( " Resultat du clustering " )
    plt.show ()
    print ("nb clusters =" ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )