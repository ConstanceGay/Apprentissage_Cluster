from scipy.io import arff
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
import hdbscan
from sklearn.datasets import make_blobs
import time
import random
import warnings
warnings.simplefilter('ignore')

# recuperation du premier dataset
d2d_4c = arff.loadarff(open('./data/2d-4c.arff', 'r'))

# Comme les données ne sont pas dans le format souhaité on les ranges
tab = d2d_4c[0]
tab_x = []          # tableau contenant les coordonnées x des points
tab_y = []          # tableau contenant les coordonnées y des points
tab_col = []        # tableau contenant les couleurs des points
array = []          # tableau contenant les coordonnées x et y des points
for i in range(0, tab.size):
    tab_x.append(tab[i][0])
    tab_y.append(tab[i][1])
    tab_col.append(tab[i][2])
    array.append([tab[i][0], tab[i][1]])

#####################################################################################
print("HDBSCAN : test sur 2d-4c avec min_cluster_size=10\n")

clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(array)

plt.scatter(tab_x, tab_y, c=cluster_labels)

#####################################################################################
# import des données
dartboard1 = arff.loadarff(open('./data/dartboard1.arff', 'r'))
diamond9 = arff.loadarff(open('./data/diamond9.arff', 'r'))
engytime = arff.loadarff(open('./data/engytime.arff', 'r'))
sizes4 = arff.loadarff(open('./data/sizes4.arff', 'r'))
cure_t2_4k = arff.loadarff(open('./data/cure-t2-4k.arff', 'r'))

datasets = [dartboard1, diamond9, engytime, sizes4, cure_t2_4k]
nom_datasets = ["dartboard1", "diamond9", "engytime", "sizes4", "cure-t2-4k"]

print("Application de HDBSCAN sur différents datasets avec min_cluster_size=10\n")

graph = []
temps = []

for j in range(0, 5):
    tab = datasets[j][0]

    tab_x = []
    tab_y = []
    tab_col = []
    array = []
    for i in range(0, tab.size):
        tab_x.append(tab[i][0])
        tab_y.append(tab[i][1])
        tab_col.append(tab[i][2])
        array.append([tab[i][0], tab[i][1]])

    begin = time.process_time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(array)
    silhouette_avg = silhouette_score(array, cluster_labels)
    graph.append(silhouette_avg)
    t = time.process_time() - begin
    temps.append(t)

    titre = nom_datasets[j]
    plt.title(titre)
    plt.scatter(tab_x, tab_y, c=cluster_labels)
    matplotlib.pyplot.show()

y_axis = [temps, graph]
y_label = ['Temps', 'Silhouette Avg']
for i in range(0, 2):
    plt.plot(nom_datasets, y_axis[i])
    plt.ylabel(y_label[i])
    plt.title(y_label[i])
    plt.show()
