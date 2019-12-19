from scipy.io import arff
import matplotlib
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
import time
import random
import warnings

warnings.simplefilter('ignore')

# recuperation du premier dataset
d2d_4c = arff.loadarff(open('./data/2d-4c.arff', 'r'))

# Comme les données ne sont pas dans le format souhaité on les ranges
tab = d2d_4c[0]
tab_x = []  # tableau contenant les coordonnées x des points
tab_y = []  # tableau contenant les coordonnées y des points
tab_col = []  # tableau contenant les couleurs des points
array = []  # tableau contenant les coordonnées x et y des points
for i in range(0, tab.size):
    tab_x.append(tab[i][0])
    tab_y.append(tab[i][1])
    tab_col.append(tab[i][2])
    array.append([tab[i][0], tab[i][1]])

#####################################################################################
print("DBSCAN : valeurs au hasard pour min-sample & eps en laissant la métrique de distance à sa valeur par défaut\n")

graph = []
temps = []
values = []

min_samplesV = random.randint(1, 10)
epsV = round(random.uniform(0.1, 5), 2)
values.append("eps=" + str(epsV) + "\n min_smp:" + str(min_samplesV))
begin = time.process_time()

clf = DBSCAN(eps=epsV, min_samples=min_samplesV)
clf.fit(array)
cluster_labels = clf.fit_predict(array)

bouldin_score = davies_bouldin_score(array, cluster_labels)
graph.append(bouldin_score)
t = time.process_time() - begin
temps.append(t)

plt.scatter(tab_x, tab_y, c=cluster_labels)
print("Values: ", values[0])
print("Temps: ", temps)
print("Score: ", graph)

#####################################################################################
print("DBSCAN : iteration de valeurs au hasard min-sample & eps \n")

graph = []
temps = []
values = []
mini = 10000
ind_min = 0

for i in range(0, 6):
    min_samplesV = random.randint(1, 10)
    epsV = round(random.uniform(0.1, 5), 2)
    values.append("eps=" + str(epsV) + "\n min_smp:" + str(min_samplesV))
    begin = time.process_time()

    clf = DBSCAN(eps=epsV, min_samples=min_samplesV)
    clf.fit(array)
    cluster_labels = clf.fit_predict(array)
    bouldin_score = davies_bouldin_score(array, cluster_labels)
    graph.append(bouldin_score)
    t = time.process_time() - begin
    temps.append(t)

    if bouldin_score < mini:
        ind_min = i
        mini = bouldin_score
        min_label = cluster_labels

print("Les meilleurs valeurs sont: ", values[ind_min])

plt.title(values[ind_min])
plt.scatter(tab_x, tab_y, c=max_color)
plt.show()

y_axis = [temps, graph]
y_label = ['Temps', 'Bouldin']
for i in range(0, 2):
    plt.plot(values, y_axis[i])
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

print("Méthode DBSCAN itérative sur eps & min_sample avec différents datasets : \n")


# fonction permettant de verifier si un tableau contient une seule et même valeur dans toutes ses cases
def check_equal(tableau):
    first_value = tableau[0]
    for k in (1, (len(tableau)) - 1):
        if tableau[k] != first_value:
            return False
    return True


##########################################

graph_global = []
temps_global = []

for j in range(0, 5):
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

    graph = []
    temps = []
    temps_max = 0
    color_max = []
    maxi = -1
    ind_max = 0
    for i in range(0, 100):
        min_samplesV = random.randint(1, 10)
        epsV = round(random.uniform(0.01, 5), 2)
        values.append("eps=" + str(epsV) + "\n min_smp:" + str(min_samplesV))

        begin = time.process_time()
        clf = DBSCAN(eps=epsV, min_samples=min_samplesV)
        clf.fit(array)
        cluster_labels = clf.fit_predict(array)

        # on filtre les cas où un seul cluster a été trouvé
        if not (check_equal(cluster_labels)):
            silhouette_avg = silhouette_score(array, cluster_labels)
            graph.append(silhouette_avg)
            t = time.process_time() - begin
            temps.append(t)
            if silhouette_avg > maxi:
                temps_max = t
                ind_max = i
                maxi = silhouette_avg
                color_max = cluster_labels
        else:
            graph.append(0)
            temps.append(0)

    graph_global.append(maxi)
    temps_global.append(temps_max)

    print("Les values les plus appropriées sont: ", values[ind_max])

    titre = nom_datasets[j] + " avec " + values[ind_max]
    plt.title(titre)
    plt.scatter(tab_x, tab_y, c=color_max)
    matplotlib.pyplot.show()

    y_axis = [temps, graph]
    y_label = ['Temps', 'Silhouette Avg']
    for i in range(0, 2):
        plt.plot(y_axis[i])
        plt.ylabel(y_label[i])
        plt.title(nom_datasets[j])
        plt.show()

y_axis = [temps_global, graph_global]
y_label = ['Temps', 'Silhouette Avg']
for i in range(0, 2):
    plt.plot(nom_datasets, y_axis[i])
    plt.ylabel(y_label[i])
    plt.title(y_label[i])
    plt.show()
