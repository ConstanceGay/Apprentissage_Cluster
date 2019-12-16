from scipy.io import arff
import matplotlib
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
import time
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

# On plot les clusters tels qu'ils sont dans le dataset
matplotlib.pyplot.scatter(tab_x, tab_y, c=tab_col)

#####################################################################################
print(" K MEANS \n")
print("Entrainement du classifieur en lui indiquant 4 clusters")

# entrainement du classifieur avec 4 clusters
clf = KMeans(n_clusters=4)
clf.fit(array)
colors = clf.predict(array)

plt.title("4 clusters")
plt.scatter(tab_x, tab_y, c=colors)

print("**************************************************\n")
print("Application itérative de la méthode précédente pour déterminer le bon nombre de clusters\n")
print("Critère d'evaluation : Silhouette average")

graph = []
temps = []
maxi = 0
ind_max = 0

for nb_clust in range(2, 8):
    begin = time.process_time()
    clf = KMeans(n_clusters=nb_clust)
    clf.fit(array)
    cluster_labels = clf.predict(array)
    silhouette_avg = silhouette_score(array, cluster_labels)
    graph.append(silhouette_avg)
    t = time.process_time() - begin
    temps.append(t)
    if silhouette_avg > maxi:
        ind_max = nb_clust
        maxi = silhouette_avg

print("Le nombre de clusters le plus approprié est : ", ind_max)
y_axis = [temps, graph]
y_label = ['Temps', 'Silhouette Avg']
for i in range(0, 2):
    plt.plot(['2', '3', '4', '5', '6', '7'], y_axis[i])
    plt.ylabel(y_label[i])
    plt.show()

print("Critère d'evaluation : Davies Bouldin")
graph = []
temps = []
mini = 1
ind_min = 0

for nb_clust in range(2, 8):
    begin = time.process_time()
    clf = KMeans(n_clusters=nb_clust)
    clf.fit(array)
    cluster_labels = clf.predict(array)
    bouldin_score = davies_bouldin_score(array, cluster_labels)
    graph.append(bouldin_score)
    t = time.process_time() - begin
    temps.append(t)
    if bouldin_score < mini:
        ind_min = nb_clust
        mini = bouldin_score

print("Le nombre de clusters le plus approprié est : ", ind_min)
y_axis = [temps, graph]
y_label = ['Temps', 'Davies_Bouldin Score']
for i in range(0, 2):
    plt.plot(['2', '3', '4', '5', '6', '7'], y_axis[i])
    plt.ylabel(y_label[i])
    plt.show()

print("**************************************************\n")
# Import de nouveaux datasets
dartboard1 = arff.loadarff(open('./data/dartboard1.arff', 'r'))
engytime = arff.loadarff(open('./data/engytime.arff', 'r'))
sizes4 = arff.loadarff(open('./data/sizes4.arff', 'r'))

datasets = [dartboard1, engytime, sizes4]
nom_datasets = ["dartboard1", "engytime", "sizes4"]
file_type = ['Non convexes', 'Convexe & mal sépararéés', 'Densité variable']

print("Méthode k-Means itérative sur le nombre de clusters avec différents datasets : \n")

for j in range(0, 3):
    # mise en forme du dataset
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

    # iteration sur le nombre de voisins
    graph = []
    temps = []
    color_max = []
    maxi = 0
    ind_max = 0
    for nb_clust in range(2, 8):
        begin = time.process_time()
        clf = KMeans(n_clusters=nb_clust)
        clf.fit(array)
        cluster_labels = clf.predict(array)
        silhouette_avg = silhouette_score(array, cluster_labels)
        graph.append(silhouette_avg)
        t = time.process_time() - begin
        temps.append(t)
        if silhouette_avg > maxi:
            ind_max = nb_clust
            maxi = silhouette_avg
            color_max = cluster_labels

    print("Le nombre de clusters le plus approprié pour ", nom_datasets[j], " est : ", ind_max)

    # representation visuelle du clustering
    titre = nom_datasets[j] + " avec " + str(ind_max) + " clusters"
    plt.title(titre)
    plt.scatter(tab_x, tab_y, c=color_max)
    matplotlib.pyplot.show()

    # graph du temps et du score
    y_axis = [temps, graph]
    y_label = ['Temps', 'Silhouette Avg']
    for i in range(0, 2):
        plt.plot(['2', '3', '4', '5', '6', '7'], y_axis[i])
        plt.ylabel(y_label[i])
        plt.title(nom_datasets[j])
        plt.show()
