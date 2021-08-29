from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from csv import reader, writer, QUOTE_NONNUMERIC
import math
import numpy as np

# read data from dataset file
def read_data(file_name):
    data = {
        'x': [],
        'y': [],
        'z': [],
        'w': []
    }
    with open(file_name+'.csv', 'r') as read_obj:
        csv_reader = reader(read_obj, quoting=QUOTE_NONNUMERIC)
        for row in csv_reader:
            data['x'].append(row[0])
            data['y'].append(row[1])
            data['z'].append(row[2])
            data['w'].append(row[3])
    return data

# write a csv file based on given parameters
def write_data(labels, result, label):
    with open(label+'.csv', 'w') as f:
        labels = np.array(labels).tolist()
        w = writer(f)
        for row in labels:
            w.writerow([row])
        w.writerow([result])

    

def calculate_d_s(data, labels_euclidean, labels_cosine):
    minimum_d = 9999999
    minimum_s = 9999999
    # for all the points
    for i in range(len(labels_euclidean)):
        for j in range(len(labels_euclidean)):
            p1 = [data['x'][i], data['y'][i], data['z'][i], data['w'][i]]
            p2 = [data['x'][j], data['y'][j], data['z'][j], data['w'][j]]
            # if points aren't in the same cluster
            if (labels_euclidean[i] != labels_euclidean[j]):
                #calculate the minimum distance
                d = distance.euclidean(p1, p2)
                if (d < minimum_d):
                    minimum_d = d
            # if points aren't in the same cluster
            if (labels_cosine[i] != labels_cosine[j]):
                #calculate the minimum distance
                s = distance.cosine(p1, p2)
                if (s < minimum_s):
                    minimum_s = s
    # return minimum distance and maximum similarity
    return minimum_d, 1 - minimum_s
        
def run():
    data = read_data('dataset')
    df = DataFrame(data, columns=['x','y','z','w'])
    # cluster with KMeans which uses euclidean distance
    labels_euclidean = KMeans(n_clusters=2).fit_predict(df)
    # cluster with Agglomerative which uses cosine distance
    labels_cosine = AgglomerativeClustering().fit(df).labels_
    # calculate d and s
    d, s = calculate_d_s(data, labels_euclidean, labels_cosine)
    # write data
    write_data(labels_euclidean, d, 'euclidean')
    write_data(labels_cosine, s, 'cosine')
    
    

run()