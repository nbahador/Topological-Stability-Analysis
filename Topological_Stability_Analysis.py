# ------------------------------------------ #
# ------------------------------------------ #
# Topological Stability Analysis
# Structural balance theory
# Three-node motif
# Four-node motif
# five-node motif
# six-node motif
# Weight of Negative Motifs
# ------------------------------------------ #
# ------------------------------------------ #
# Code writer: Nooshin Bahador
# ------------------------------------------ #
# ------------------------------------------ #

import numpy as np
from numpy.linalg import inv

## ------------------------------------------------------------------------------ ##
## Defining the network, Node features, Labels for nodes' cluster
## ------------------------------------------------------------------------------ ##

delta = 1


## ------------------------------------------------------------------------------ ##
# numerical example
# https://hal.archives-ouvertes.fr/hal-02867840/document
## ------------------------------------------------------------------------------ ##

b10 = 0.1295
a11 = 517.0544
b11 = 115.5967
a12 = 4.2614
b12 = 4.6361
a13 = 1.3083
b13 = 1.4428
a14 = 2.7480
b14 = 3.1052
a21 = 0.9492
b20 = -0.5262
a22 = 2.6331
b21 = 3.7399
a41 = 239.0092
b22 = 9.6729
a42 = 1.7819
b40 = 0.1595
a43 = 3.6549
b41 = 26.8926
a44 = 5.2346
b42 = 1.9330
a31 = 191.8224
b43 = 4.0660
a32 = 49.4899
b44 = 5.9926
b31 = 0.9688
b30 = 3.3
b32 = 0.2043


adj_matrix_0 = np.array([[0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1],
                       [-a14,a13,-a12,-a11,0,0,-a22,-a21],
                       [0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1],
                       [0,0,-a32,-a31,-a44,-a43,-a42,-a41]])

adj_matrix = np.array([[0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1],
                       [-a14,a13,-a12,-a11,0,0,-a22,-a21],
                       [0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1],
                       [0,0,-a32,-a31,-a44,-a43,-a42,-a41]])




beta = 1   # perturbation factor

fea_matrix = np.array(([0.5*beta,-0.1*beta,0.3*beta],
                       [0.2,0.1,0.7],
                       [-0.5,0.7,-0.1],
                       [-0.1,-0.6,0.4],
                       [0.3,-0.5,-0.2],
                       [0.1,-0.1,-0.4],
                       [0.3,0.8,-0.1],
                       [0.1,-0.2,0.2]), dtype=float)




Target_output = np.array(([0.01],[0.2],[0.2],[0.01],[0.01],[0.2],[0.2],[0.01]), dtype=float)


## ------------------------------------------------------------------------------ ##
## Plotting the topology of the network
## ------------------------------------------------------------------------------ ##
import matplotlib.pyplot as plt
import networkx as nx
G = nx.DiGraph()

for i in range(len(adj_matrix_0[1,:])):
 for j in range(len(adj_matrix_0[:,1])):
   if adj_matrix_0[i,j] != 0:
       G.add_edge(i, j, weight=adj_matrix_0[i,j])
   if adj_matrix_0[j,i] != 0:
       G.add_edge(j,i,weight=adj_matrix_0[j,i])



pos = nx.spiral_layout(G) #nx.circular_layout(G)
nx.draw(G,pos=pos, with_labels = True )
edgeLabels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = edgeLabels)
#plt.show()



A = adj_matrix_0
num_iter = 8
s111 = (num_iter, 1)
sum_weights_total_2 = np.zeros(s111)


# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# Three-node motif
num_iter = 8
s111 = (num_iter, 1)
count_positive_total_Three_node = np.zeros(s111)
count_negative_total_Three_node = np.zeros(s111)

for initial_node in range(len(adj_matrix_0[1,:])): # Starting node
    ax = plt.gca()
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_nodes(G, pos, nodelist=[initial_node], node_color='r')
    ax.set_title('Start walking from this node')
    #plt.show()

    Aprim = adj_matrix_0
    num_iter = 8
    s111 = (num_iter, 1)
    sum_weights_total = np.zeros(s111)

    num_iter = 1
    s111 = (num_iter, 4)
    path_total = np.zeros(s111)

    num_iter = 1
    s111 = (num_iter, 1)
    sum_total = np.zeros(s111)

    count_positive = 0
    count_negative = 0

    for n in range(len(adj_matrix_0[1,:])):
        previous_node = initial_node
        current_node = n

        for m in range(len(adj_matrix_0[1,:])):
            next_node = m


            for g in range(len(adj_matrix_0[1,:])):
                last_node = g


                if (previous_node != current_node) and (current_node != next_node) and (next_node != last_node)  \
                        and (last_node == previous_node) and (A[previous_node, current_node] != 0) and  \
                        (A[current_node, next_node] != 0) and (A[next_node, last_node] != 0):

                    ax = plt.gca()

                    ax.set_title('Starting Node = ' + str(initial_node))
                    nx.draw(G, pos, with_labels=True, ax=ax)
                    path = [initial_node, current_node, next_node , last_node]
                    path_edges = list(zip(path, path[1:]))
                    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r')
                    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=10)
                    #plt.show()

                    S1 = np.sign(adj_matrix_0[initial_node, current_node])
                    P1 = adj_matrix_0[initial_node, current_node]
                    S2 = np.sign(adj_matrix_0[current_node, next_node])
                    P2 = adj_matrix_0[current_node, next_node]
                    S3 = np.sign(adj_matrix_0[next_node, last_node])
                    P3 = adj_matrix_0[next_node, last_node]
                    S_total = S1 * S2 * S3
                    P_total = P1 * P2 * P3

                    if np.sign(S_total)>0:
                        count_positive += P_total
                    elif np.sign(S_total)<0:
                        count_negative += P_total
                    #print(P1 , P2 , P3)
                    #print(P_total)


    if not count_positive:
        count_positive = 0
    if not count_negative:
        count_negative = 0
    count_positive_total_Three_node[initial_node] = count_positive/(G.degree[initial_node]**2)
    count_negative_total_Three_node[initial_node] = count_negative/(G.degree[initial_node]**2)


                    #sum_total = np.append(sum_total, np.array(sum_weights).reshape([-1, 1]).T, axis=0)


# ------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------- #
# Four-node motif
num_iter = 8
s111 = (num_iter, 1)
count_positive_total_Four_node = np.zeros(s111)
count_negative_total_Four_node = np.zeros(s111)

for initial_node in range(len(adj_matrix_0[1,:])): # Starting node
    ax = plt.gca()
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_nodes(G, pos, nodelist=[initial_node], node_color='r')
    ax.set_title('Start walking from this node')
    #plt.show()

    Aprim = nx.to_numpy_array(G)
    num_iter = 8
    s111 = (num_iter, 1)
    sum_weights_total = np.zeros(s111)

    num_iter = 1
    s111 = (num_iter, 5)
    path_total = np.zeros(s111)

    num_iter = 1
    s111 = (num_iter, 1)
    sum_total = np.zeros(s111)

    count_positive = 0
    count_negative = 0

    for n in range(len(adj_matrix_0[1,:])):
        previous_node = initial_node
        current_node = n

        for m in range(len(adj_matrix_0[1,:])):
            next_node = m


            for g in range(len(adj_matrix_0[1,:])):
                next_node_2 = g


                for f in range(len(adj_matrix_0[1, :])):
                    last_node = f


                    if (previous_node != current_node) and (current_node != next_node) and (next_node != next_node_2) and (next_node_2 != last_node) \
                            and (last_node == previous_node) and (A[previous_node, current_node] != 0) and \
                            (A[current_node, next_node] != 0) and (A[next_node, next_node_2] != 0) and (A[next_node_2, last_node] != 0) and \
                            (previous_node != next_node) and (current_node != next_node_2):

                        ax = plt.gca()
                        ax.set_title('Starting Node = ' + str(initial_node))
                        nx.draw(G, pos, with_labels=True, ax=ax)
                        path = [initial_node, current_node, next_node, next_node_2, last_node]
                        #print(path)
                        path_edges = list(zip(path, path[1:]))
                        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r')
                        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=10)
                        #plt.show()

                        xx = path_total
                        yy = np.array(path).reshape([-1, 1]).T
                        path_total = np.append(xx, yy, axis=0)

                        S1 = np.sign(adj_matrix_0[initial_node, current_node])
                        P1 = adj_matrix_0[initial_node, current_node]
                        S2 = np.sign(adj_matrix_0[current_node, next_node])
                        P2 = adj_matrix_0[current_node, next_node]
                        S3 = np.sign(adj_matrix_0[next_node, next_node_2])
                        P3 = adj_matrix_0[next_node, next_node_2]
                        S4 = np.sign(adj_matrix_0[next_node_2, last_node])
                        P4 = adj_matrix_0[next_node_2, last_node]
                        S_total = S1 * S2 * S3 * S4
                        P_total = P1 * P2 * P3 * P4

                        if np.sign(S_total) > 0:
                            count_positive += P_total
                        elif np.sign(S_total) < 0:
                            count_negative += P_total

    if not count_positive:
        count_positive = 0
    if not count_negative:
        count_negative = 0
    count_positive_total_Four_node[initial_node] = count_positive/(G.degree[initial_node]**2)
    count_negative_total_Four_node[initial_node] = count_negative/(G.degree[initial_node]**2)




# ------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------- #
# five-node motif
num_iter = 8
s111 = (num_iter, 1)
count_positive_total_five_node = np.zeros(s111)
count_negative_total_five_node = np.zeros(s111)

for initial_node in range(len(adj_matrix_0[1,:])): # Starting node
    ax = plt.gca()
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_nodes(G, pos, nodelist=[initial_node], node_color='r')
    ax.set_title('Start walking from this node')
    #plt.show()

    Aprim = nx.to_numpy_array(G)
    num_iter = 8
    s111 = (num_iter, 1)
    sum_weights_total = np.zeros(s111)

    num_iter = 1
    s111 = (num_iter, 6)
    path_total = np.zeros(s111)

    num_iter = 1
    s111 = (num_iter, 1)
    sum_total = np.zeros(s111)

    count_positive = 0
    count_negative = 0

    for n in range(len(adj_matrix_0[1,:])):
        previous_node = initial_node
        current_node = n

        for m in range(len(adj_matrix_0[1,:])):
            next_node = m


            for g in range(len(adj_matrix_0[1,:])):
                next_node_2 = g


                for f in range(len(adj_matrix_0[1, :])):
                    next_node_3 = f


                    for h in range(len(adj_matrix_0[1, :])):
                        last_node = h


                        if (previous_node != current_node) and (current_node != next_node) and (
                                next_node != next_node_2) and (next_node_2 != next_node_3) and (next_node_3 != last_node) \
                                and (last_node == previous_node) and (A[previous_node, current_node] != 0) and \
                                (A[current_node, next_node] != 0) and (A[next_node, next_node_2] != 0) and (
                                A[next_node_2, next_node_3] != 0) and (A[next_node_3, last_node] != 0) and \
                                (previous_node != next_node_2) and (current_node != last_node) \
                                and (current_node != next_node_3) and (next_node != last_node) \
                                and (next_node != next_node_3) and (current_node != next_node_2) \
                                and (next_node_2 != last_node) and (next_node_3 != previous_node) and (previous_node != next_node):

                            ax = plt.gca()
                            ax.set_title('Starting Node = ' + str(initial_node))
                            nx.draw(G, pos, with_labels=True, ax=ax)
                            path = [initial_node, current_node, next_node, next_node_2, next_node_3, last_node]
                            #print(path)
                            path_edges = list(zip(path, path[1:]))
                            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r')
                            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=10)
                            #plt.show()

                            xx = path_total
                            yy = np.array(path).reshape([-1, 1]).T
                            path_total = np.append(xx, yy, axis=0)

                            S1 = np.sign(adj_matrix_0[initial_node, current_node])
                            P1 = adj_matrix_0[initial_node, current_node]
                            S2 = np.sign(adj_matrix_0[current_node, next_node])
                            P2 = adj_matrix_0[current_node, next_node]
                            S3 = np.sign(adj_matrix_0[next_node, next_node_2])
                            P3 = adj_matrix_0[next_node, next_node_2]
                            S4 = np.sign(adj_matrix_0[next_node_2, next_node_3])
                            P4 = adj_matrix_0[next_node_2, next_node_3]
                            S5 = np.sign(adj_matrix_0[next_node_3, last_node])
                            P5 = adj_matrix_0[next_node_3, last_node]
                            S_total = S1 * S2 * S3 * S4 * S5
                            P_total = P1 * P2 * P3 * P4 * P5
                            #print(P_total)

                            if np.sign(S_total) > 0:
                                count_positive += P_total
                            elif np.sign(S_total) < 0:
                                count_negative += P_total

    if not count_positive:
        count_positive = 0
    if not count_negative:
        count_negative = 0
    count_positive_total_five_node[initial_node] = count_positive/(G.degree[initial_node]**2)
    count_negative_total_five_node[initial_node] = count_negative/(G.degree[initial_node]**2)



# ------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------- #
# six-node motif
num_iter = 8
s111 = (num_iter, 1)
count_positive_total_six_node = np.zeros(s111)
count_negative_total_six_node = np.zeros(s111)

for initial_node in range(len(adj_matrix_0[1,:])): # Starting node
    ax = plt.gca()
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_nodes(G, pos, nodelist=[initial_node], node_color='r')
    ax.set_title('Start walking from this node')
    #plt.show()

    Aprim = nx.to_numpy_array(G)
    num_iter = 8
    s111 = (num_iter, 1)
    sum_weights_total = np.zeros(s111)

    num_iter = 1
    s111 = (num_iter, 7)
    path_total = np.zeros(s111)

    num_iter = 1
    s111 = (num_iter, 1)
    sum_total = np.zeros(s111)

    count_positive = 0
    count_negative = 0

    for n in range(len(adj_matrix_0[1,:])):
        previous_node = initial_node
        current_node = n

        for m in range(len(adj_matrix_0[1,:])):
            next_node = m


            for g in range(len(adj_matrix_0[1,:])):
                next_node_2 = g


                for f in range(len(adj_matrix_0[1, :])):
                    next_node_3 = f


                    for h in range(len(adj_matrix_0[1, :])):
                        next_node_4 = h


                        for k in range(len(adj_matrix_0[1, :])):
                            last_node = k


                            if (previous_node != current_node) and (current_node != next_node) and \
                                    (next_node != next_node_2) and (next_node_2 != next_node_3) and \
                                    (next_node_3 != next_node_4) and (next_node_4 != last_node) \
                                    and (last_node == previous_node) and (A[previous_node, current_node] != 0) and \
                                    (A[current_node, next_node] != 0) and (A[next_node, next_node_2] != 0) and \
                                    (A[next_node_2, next_node_3] != 0) and (A[next_node_3, next_node_4] != 0) and (A[next_node_4, last_node] != 0) and \
                                    (previous_node != next_node_2) and (current_node != last_node) \
                                    and (current_node != next_node_3) and (next_node != last_node) \
                                    and (next_node != next_node_3) and (current_node != next_node_2) \
                                    and (next_node_2 != last_node) and (next_node_3 != previous_node) and \
                                    (previous_node != next_node) and \
                                    (next_node_2 != next_node_4) and (next_node_3 != last_node) and (previous_node != next_node_4) and (next_node != next_node_4) and \
                                    (current_node != next_node_4) and (last_node != next_node):

                                ax = plt.gca()
                                ax.set_title('Starting Node = ' + str(initial_node))
                                nx.draw(G, pos, with_labels=True, ax=ax)
                                path = [initial_node, current_node, next_node, next_node_2, next_node_3, next_node_4, last_node]
                                # print(path)
                                path_edges = list(zip(path, path[1:]))
                                nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r')
                                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=10)
                                # plt.show()

                                xx = path_total
                                yy = np.array(path).reshape([-1, 1]).T
                                path_total = np.append(xx, yy, axis=0)

                                S1 = np.sign(adj_matrix_0[initial_node, current_node])
                                P1 = adj_matrix_0[initial_node, current_node]
                                S2 = np.sign(adj_matrix_0[current_node, next_node])
                                P2 = adj_matrix_0[current_node, next_node]
                                S3 = np.sign(adj_matrix_0[next_node, next_node_2])
                                P3 = adj_matrix_0[next_node, next_node_2]
                                S4 = np.sign(adj_matrix_0[next_node_2, next_node_3])
                                P4 = adj_matrix_0[next_node_2, next_node_3]
                                S5 = np.sign(adj_matrix_0[next_node_3, next_node_4])
                                P5 = adj_matrix_0[next_node_3, next_node_4]
                                S6 = np.sign(adj_matrix_0[next_node_4, last_node])
                                P6 = adj_matrix_0[next_node_4, last_node]
                                S_total = S1 * S2 * S3 * S4 * S5 * S6
                                P_total = P1 * P2 * P3 * P4 * P5 * P6
                                # print(P_total)

                                if np.sign(S_total) > 0:
                                    count_positive += P_total
                                elif np.sign(S_total) < 0:
                                    count_negative += P_total

    if not count_positive:
        count_positive = 0
    if not count_negative:
        count_negative = 0
    count_positive_total_six_node[initial_node] = count_positive/(G.degree[initial_node]**2)
    count_negative_total_six_node[initial_node] = count_negative/(G.degree[initial_node]**2)


count_negative_graph = count_negative_total_Three_node*count_negative_total_Four_node*count_negative_total_five_node*count_negative_total_six_node
count_negative_graph = abs(count_negative_graph) ** (1. / 3)

count_positive_graph = count_positive_total_Three_node*count_positive_total_Four_node*count_positive_total_five_node*count_positive_total_six_node
count_positive_graph = abs(count_positive_graph) ** (1. / 3)



plt.close('all')

objects = ('Node 0', 'Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5', 'Node 6', 'Node 7')
y_pos = np.arange(len(objects))
performance = count_negative_graph.squeeze()
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Total Cost Associated With \n Imbalanced Motif-Paths Traversed From Each Node')
plt.xlabel('Starting Node')
plt.show()



















