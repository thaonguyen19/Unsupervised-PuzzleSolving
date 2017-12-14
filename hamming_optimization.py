import random
import numpy as np
import sklearn.cluster
from scipy.spatial.distance import hamming

def generate_all_permutations():
    all_permutations = []
    count = 0
    while count < 64:
        l = list(range(1, 10))
        random.shuffle(l)
        if l not in all_permutations:
            all_permutations.append(l)
            count += 1
    return all_permutations


def find_ave_hamming(all_permutations_arranged):
    ave_hamming = []
    for i in range(1, len(all_permutations_arranged)):
        ave_hamming.append(hamming(all_permutations_arranged[i-1], all_permutations_arranged[i]))
    return float(sum(ave_hamming))/len(ave_hamming)

def rearrange_cluster(all_permutations):
    all_permutations_arranged = []
    lev_similarity = -1.0*np.array([[hamming(w1,w2) for w1 in all_permutations] for w2 in all_permutations])
    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(lev_similarity)

    for cluster_id in np.unique(affprop.labels_):
        #exemplar = all_permutations[affprop.cluster_centers_indices_[cluster_id]]
        all_strs_in_cluster = []
        for ind in np.nonzero(affprop.labels_==cluster_id)[0]:
            all_strs_in_cluster.append(all_permutations[ind])

        all_permutations_arranged.extend(all_strs_in_cluster)
    assert len(all_permutations_arranged) == len(all_permutations)
    return all_permutations_arranged 

def rearrange_greedy(all_permutations):
    all_permutations_arranged = []
    all_permutations_copied = all_permutations[:]
    center = random.choice(all_permutations_copied)
    all_permutations_arranged.append(center)
    all_permutations_copied.remove(center)
    left = random.choice(all_permutations_copied)
    all_permutations_copied.remove(left)
    all_permutations_arranged.insert(0, left)
    right = random.choice(all_permutations_copied)
    all_permutations_copied.remove(right)
    all_permutations_arranged.append(right)
    while len(all_permutations_copied) > 0:
        curr = random.choice(all_permutations_copied)
        all_permutations_copied.remove(curr)
        if hamming(curr, left) < hamming(curr, right):
            #left_ind = 0#all_permutations_copied.index(left)
            if hamming (all_permutations_arranged[1], left) < hamming(curr, all_permutations_arranged[1]):
                left = curr
                all_permutations_arranged.insert(0, curr)
            else:
                all_permutations_arranged.insert(1, curr)
        else:
            if hamming (all_permutations_arranged[len(all_permutations_arranged)-2], right) < hamming(curr, all_permutations_arranged[len(all_permutations_arranged)-2]):
                right = curr
                all_permutations_arranged.append(curr)
            else:
                all_permutations_arranged.insert(len(all_permutations_arranged)-1, curr)

    assert len(all_permutations_arranged) == len(all_permutations)
    return all_permutations_arranged

if __name__ == '__main__':
    all_permutations = generate_all_permutations()
    print "RANDOM HAMMING: ", find_ave_hamming(all_permutations)
    all_permutations_arranged = rearrange_cluster(all_permutations)
    print "CLUSTERING HAMMING: ", find_ave_hamming(all_permutations_arranged)
    all_permutations_arranged = rearrange_greedy(all_permutations)
    hamming_value = find_ave_hamming(all_permutations_arranged)
    while hamming_value > 0.77:
        all_permutations_arranged = rearrange_greedy(all_permutations)
        hamming_value = find_ave_hamming(all_permutations_arranged)
    print "GREEDY HAMMING: ", hamming_value

