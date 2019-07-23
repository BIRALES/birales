import numpy as np
from scipy.spatial import KDTree, Rectangle


def get_clusters(ndx, c_labels):
    # Add cluster labels to the data
    labelled_data = np.append(ndx, np.expand_dims(c_labels, axis=1), axis=1)

    # Cluster mask to remove noise clusters
    de_noised_data = labelled_data[labelled_data[:, 3] > -1]

    de_noised_data = de_noised_data[de_noised_data[:, 3].argsort()]

    # Return the location at which clusters where identified
    cluster_ids = np.unique(de_noised_data[:, 3], return_index=True)

    # Split the data into clusters
    clusters = np.split(de_noised_data, cluster_ids[1])

    # remove empty clusters
    clusters = [x for x in clusters if np.any(x)]

    return clusters
