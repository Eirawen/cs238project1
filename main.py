
import numpy as np
from scipy.special import gamma
import pandas as pd
from itertools import product
from csv import reader
import random


def node_score (data, child, parents):
    #computes given score of node for k2 algorithm
    #data is a np array of the data set
    #child is the index of the child node
    #parents are a list[int] of column indices for parent node
    #returns a bayesian score for the node configuration

    r = len(np.unique(data[:, child]))

    parent_combinations = product(*[np.unique(data[:, p]) for p in parents])

    score = 0
    alpha = 1 
    #uniform dirichlet prior

    for parents in parent_combinations:
        if not parents:
            parent_data = data
        else:
            parent_indices = np.array(parents)
            parent_values = np.array(parents)
            mask = np.all(data[:, parent_indices] == parent_values, axis=1)
            parent_data = mask

        m_ij0 = len(parent_data)

        score += np.log(gamma(alpha) / gamma(alpha + m_ij0))
        for i in range(r):

            # Count of child values given this parent combination
            m_ijk = np.sum(parent_data[:, child] == i + 1)
            if m_ijk < 0:
                print("why is m_ijk negative")
            score += np.log(gamma(alpha + m_ijk) / gamma(alpha))

    return score

def k2_algo(data, node_order, max_Parents = 3):
    #data is np array of dataset
    #max_parents is max parents for any node.
    n = data.shape[1]
    G = np.zeros((n, n), dtype = int)
    score_total = 0 


    for index, child in enumerate(node_order):

        p_old = node_score(data, child, [])
        score_total += p_old
        prior_nodes = node_order[:index]
        current_parents = []


        while prior_nodes and np.sum(G[child]) < max_Parents:
            scores = [node_score(data, child, current_parents + [p]) for p in prior_nodes]
            max_index = np.argmax(scores)
            p_new = scores[max_index]

            if p_new > p_old:
                p_old = p_new
                top_parent = prior_nodes[max_index]
                G[child, top_parent] = 1
                current_parents.append(top_parent)
                prior_nodes.remove(top_parent)
            else:
                break
    return G, score_total

def score_maximizer(csv_path, iterations, max_parents):
    try:
        data = pd.read_csv(csv_path)
    except:
        print("Error: CSV file not found")
        return
    csv_headers = list(data.columns)
    best_score = -1000000
    best_graph = None

    for i in range(iterations):
        random.shuffle(csv_headers)
        node_order = [csv_headers.index(h) for h in csv_headers]
        current_graph, current_score = k2_algo(data.values, node_order, max_parents)

        if current_score > best_score:
            best_score = current_score
            best_graph = current_graph
    
    return best_graph, best_score

def write_gph(G, csv_path, output_file, best_score):
    headers = None
    with open(csv_path, 'r') as f:
        headers = f.readline().strip().split(',')
    
    with open(output_file, 'w') as f:
        for i in range(G.shape[0] - 1):
            for j in range(G.shape[1]):
                if G[i][j] == 1:
                    f.write(f"{headers[i]}, {headers[j]}\n")
        f.write(f"Score: {best_score}")
def main():
    csv_path = "small.csv"
    output_file = "output.gph"
    iterations = 3
    max_parents = 3
    best_graph, best_score = score_maximizer(csv_path, iterations, max_parents)
    print(best_graph)
    print(best_score)
    
    write_gph(best_graph, csv_path, output_file, best_score)
    



if __name__ == "__main__":
    main()

