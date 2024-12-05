# Written by Eva Powlison

import pandas as pd
import sys
import os
from os import path
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


CONSOLE_COLOR_RED = "\033[91m"
CONSOLE_COLOR_MAG = "\033[95m"
CONSOLE_COLOR_RESET = "\033[39m"

COLS_OF_INTEREST = [
    #0, # Reporting Railroad Code
    2, # Report Year
    5, # Accident Year
    6, # Accident Month
    #7, # Other Railroad Code
    #12, # Maintenance Railroad Code
    #30, # Station name
    #33, # State Abbreviation
    21, # Accident Type
    41, # Weather Condition Code
    54, # Train Speed
    #57, # Gross Tonnage
    59, # Signalization
]

total_to_process = 0
done_processing = 0

def updateStatus():
    global done_processing
    done_processing += 1
    percentage: int = int((done_processing / total_to_process) * 100)
    sys.stdout.write("\r")
    sys.stdout.write(f"Clustering {percentage}% complete")
    sys.stdout.flush()

def preprocessData(df, verbose):
    if (verbose == True):
       print(df.info())
       print(df.describe())

       for col in df.columns:
           unique = len(df[col].unique())
           print(f"{col} has {unique} values")

    df = df.fillna(0)
    encoded_data = pd.get_dummies(df)

    print("\nDataframe data one-hot encoded\n")
    
    if (verbose == True):
        print("Memory usage:")
        memory = encoded_data.memory_usage()
        print(memory)

    return encoded_data

def reduceDimensionality(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    print(f"Reduced data to {n_components} dimensions")
    return reduced_data

def dbscanClustering(data):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(data)
    print(f"DBSCAN produced {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    return labels

def findClusterSilhouettes(df, verbose):
    global done_processing
    global total_to_process

    done_processing = -1
    total_to_process = 8

    silhouette = []
    print(CONSOLE_COLOR_MAG, "\nFinding cluster silhouettes... This may take a while", CONSOLE_COLOR_RESET)
    updateStatus()
    for n in range(2,11):
        clusters = KMeans(n_clusters = n).fit(df)
        labels = clusters.fit_predict(df)
        silhouette_average = silhouette_score(df, labels)
        silhouette.append(silhouette_average)
        updateStatus()

    plt.figure(figsize=(10,6))
    plt.plot(range(2,11), silhouette, marker='o')
    plt.title("Silhouette Analysis")
    plt.xlabel("n_clusters")
    plt.ylabel("Silhouette Score")
    plt.savefig("clusterSilhouettes.png")
    plt.show()

def printHelp():
    print("This program finds clusters in a dataset containing railway incident data")
    print("Place your data in 'data/data.csv' or provide an alternative path via command line arguments")

    print("\nOptions:\n")
    print("\t -d \t Specify alternate datapath")
    print("\t -h \t Display this information")
    print("\t -n \t Specify number of clusters to use for cluster analysis")
    print("\t -v \t Display verbose info")

    print("\nDefault usage:\n")
    print("\tpython clusterAnalysis.py")

    print("\nAlternative path:\n")
    print("\tpython clusterAnalysis.py -d <path>\n")

def main():
    datapath = "data.csv"

    verbose = False
    n_clusters = 4

    if len(sys.argv) > 1:
        if "-h" in sys.argv:
            printHelp()
            exit(0)
        
        if "-d" in sys.argv:
            index = sys.argv.index("-d")
            datapath = sys.argv[index + 1]

        if "-n" in sys.argv:
            index = sys.argv.index("-n")
            n_clusters = sys.argv[index + 1]

        if "-v" in sys.argv:
            verbose = True

    if not path.exists(datapath):
        print(CONSOLE_COLOR_RED, f"Error: Unable to find data file '{datapath}'\n", CONSOLE_COLOR_RESET)
        exit(1) 

    try:
        df = pd.read_csv(datapath, usecols=COLS_OF_INTEREST)
    except FileNotFoundError:
        print(CONSOLE_COLOR_RED, "Error: file '{datapath}' not found\n", CONSOLE_COLOR_RESET)
        printHelp()
        exit(1)
    except pd.errors.EmptyDataError:
        print(CONSOLE_COLOR_RED, f"Error: data file '{datapath}' is empty\n", CONSOLE_COLOR_RESET)
        exit(1)
    
    print(CONSOLE_COLOR_MAG, f"\nData loaded with shape {df.shape}", CONSOLE_COLOR_RESET)

    processedData = preprocessData(df, verbose)
    # reducedData = reduceDimensionality(processedData)

    findClusterSilhouettes(processedData, verbose)
    # dbscanLabels = dbscanClustering(reducedData)

if __name__ == "__main__":
    main()
