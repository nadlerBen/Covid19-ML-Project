import math
import numpy as np
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
from scipy.special import comb
p_c = {'Young':0.005,'Adult':0.01,'MidAge':0.3,'Elder':0.6}
class ModelData:
    def __init__(self, dataset):
        """Expects data set file with index column (train and test) """
        self.init_graph = pd.read_csv(dataset)
        self.nodes = self.init_graph[['node1', 'node2']]
        self.Relation = self.init_graph['Relation']
def create_undirected_graph(links_data, commnity,question):
    G = nx.from_pandas_edgelist(links_data.init_graph, 'node1', 'node2', edge_attr=True, create_using=nx.Graph())
    people = list(commnity['Node'])
    age = list(commnity['Age'])
    status = list(commnity['Status{}'.format(question)])
    nodes_status_dict = dict()
    nodes_age_dict = dict()
    for i in range(len(people)):
        nodes_status_dict[people[i]] = status[i]
        nodes_age_dict[people[i]] = age[i]
    for node in G.nodes():
        if node not in people:
            G.nodes[node]['Status{}'.format(question)] = 'S'
            G.nodes[node]['Age'] = 'Adult'
    nx.set_node_attributes(G,nodes_status_dict,'Status{}'.format(question))
    nx.set_node_attributes(G,nodes_age_dict, 'Age')
    #################################### mitigation strategy 1: quarntine elders ###########################################
    counter = 0
    # for node in G.nodes():
    #   if G.nodes[node]['Age'] == 'Elder':
    #       G.nodes[node]['Status{}'.format(question)] = 'Q'
    #       counter += G.degree(node)
    #       if counter >= 25234:
    #         break
    # print(counter)
    #################################### end mitigation strategt ###########################################################
    infectious = dict()
    hosiptalized = dict()
    exposed = dict()
    sub_clinical = dict()
    recovered = dict()
    for node in G.nodes():
        if G.nodes[node]['Status{}'.format(question)] == 'I':
            infectious[node] = 2
        elif G.nodes[node]['Status{}'.format(question)] == 'Ic':
            hosiptalized[node] = 6
        elif G.nodes[node]['Status{}'.format(question)] == 'E':
            exposed[node] = 4
        elif G.nodes[node]['Status{}'.format(question)] == 'Is':
            sub_clinical[node] = 6
        elif G.nodes[node]['Status{}'.format(question)] == 'R':
            recovered[node] = 'R'
    return G,exposed,infectious,sub_clinical,hosiptalized,recovered
def daily_update():
    P_relation = {'Family': 0, 'Friends': 0, 'Work': 0}
    R0 = np.random.normal(3,1,1)
    daily_number = R0/2
    p = daily_number/21
    P_relation['Family'] = 3*p[0]
    P_relation['Work'] = p[0]
    P_relation['Friends'] = p[0]/2
    return R0[0],P_relation
def Stochastic_update(G,exposed,infectios,sub_clinical,hosiptalized,recovered,P_relation,question):
        for key,value in infectios.items():
            infectios[key] -= 1
            if key not in G.nodes():
                continue
            neighbors = G.neighbors(key)
        ##################################### UPDATE EXPOSED IN SEIR ######################
            for node in neighbors:
                if G.nodes[node]['Status{}'.format(question)] == 'S':
                    if G[node][key]['Relation'] == 'Family':
                        P = random.random()
                        if P <= P_relation['Family']:
                            G.nodes[node]['Status{}'.format(question)] = 'E'
                            exposed[node] = 5
                    elif G[node][key]['Relation'] == 'Friends':
                        P = random.random()
                        if P <= P_relation['Friends']:
                            G.nodes[node]['Status{}'.format(question)] = 'E'
                            exposed[node] = 5
                    elif G[node][key]['Relation'] == 'Work':
                        P = random.random()
                        if P <= P_relation['work']:
                            G.nodes[node]['Status{}'.format(question)] = 'E'
                            exposed[node] = 5
        ##################################### UPDATE Ic and Is IN SEIR ######################
        for node in infectios.keys():
            if node not in G.nodes():
                continue
            if infectios[node] == 0:
                P = random.random()
                if G.nodes[node]['Age'] == 'Young':
                    if P < p_c['Young']:
                        G.nodes[node]['Status{}'.format(question)] = 'Ic'
                        hosiptalized[node] = 14
                        continue
                    else:
                        G.nodes[node]['Status{}'.format(question)] = 'Is'
                        sub_clinical[node] = 7
                        continue
                elif G.nodes[node]['Age'] == 'Adult':
                    if P < p_c['Adult']:
                        G.nodes[node]['Status{}'.format(question)] = 'Ic'
                        hosiptalized[node] = 14
                        continue
                    else:
                        G.nodes[node]['Status{}'.format(question)] = 'Is'
                        sub_clinical[node] = 7
                        continue
                elif G.nodes[node]['Age'] == 'MidAge':
                    if P < p_c['MidAge']:
                        G.nodes[node]['Status{}'.format(question)] = 'Ic'
                        hosiptalized[node] = 14
                        continue
                    else:
                        G.nodes[node]['Status{}'.format(question)] = 'Is'
                        sub_clinical[node] = 7
                        continue
                elif G.nodes[node]['Age'] == 'Elder':
                    if P < p_c['Elder']:
                        G.nodes[node]['Status{}'.format(question)] = 'Ic'
                        hosiptalized[node] = 14
                        continue
                    else:
                        G.nodes[node]['Status{}'.format(question)] = 'Is'
                        sub_clinical[node] = 7
                        continue
        ####################################### Update infectious and recoverd #####################
        for node in exposed.keys():
            exposed[node] -= 1
            if exposed[node] == 0:
                G.nodes[node]['Status{}'.format(question)] = 'I'
                infectios[node] = 2
        for node in sub_clinical.keys():
            sub_clinical[node] -= 1
            if sub_clinical[node] == 0:
                G.nodes[node]['Status{}'.format(question)] = 'R'
                recovered[node] = 'R'
        for node in hosiptalized.keys():
            hosiptalized[node] -= 1
            if hosiptalized[node] == 0:
                G.nodes[node]['Status{}'.format(question)] = 'R'
                recovered[node] = 'R'
        return exposed,infectios,sub_clinical,hosiptalized,recovered
def remove_from_dict(exposed,infectios,sub_clinical,hosiptalized,recovered):
    keys_list = list()
    #################################### update infectious #########################################
    for key,value in infectios.items():
        if value == 0:
            keys_list.append(key)
    for key in keys_list:
        del infectios[key]
    #################################### update exposed ############################################
    keys_list = list()
    for key,value in exposed.items():
        if value == 0:
            keys_list.append(key)
    for key in keys_list:
        del exposed[key]
    #################################### update Ic #################################################
    keys_list = list()
    for key,value in hosiptalized.items():
        if value == 0:
            keys_list.append(key)
    for key in keys_list:
        del hosiptalized[key]
    #################################### update Is #################################################
    keys_list = list()
    for key,value in sub_clinical.items():
        if value == 0:
            keys_list.append(key)
    for key in keys_list:
        del sub_clinical[key]
    return exposed,infectios,sub_clinical,hosiptalized,recovered
def plot_graph(x,y_exposed,y_infectious,y_subclinical,y_hospitelized):
    # plotting the line 1 points
    plt.plot(x, y_exposed, label="exposed")
    # line 2 points
    # plotting the line 2 points
    plt.plot(x, y_infectious, label="infectious")
    plt.plot(x, y_subclinical, label="subclinical")
    plt.plot(x, y_hospitelized, label="hospitalized")
    #plt.plot(x, y_recoverd, label="recovered")
    plt.xlabel('x - day')
    # Set the y axis label of the current axis.
    plt.ylabel('y - number of instances')
    # Set a title of the current axes.
    plt.title('virus SEIR no interventions ')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()
    return
def plot_confidance_graph(days,dict_status,x,color,title):
    ############################## converting data into right stracture ###############################################
        list_of_elements = []
        for key,value in dict_status.items():
            days_list = []
            for val in value.values():
                days_list.append(val)
            list_of_elements.append(days_list)
        matrix = np.matrix(list_of_elements)
        upper = []
        lower = []
        y = []
        for i in range(days):
            array = np.array(matrix[:, i])
            y.append(np.mean(array))
            upper.append(np.percentile(array,80))
            lower.append(np.percentile(array,20))
        print(upper[days-1])
        print(y[days-1])
        print(lower[days-1])
        ######################### plot graph ##########################################################################
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, lower, upper, color=color, alpha=.1)
        plt.xlabel('x - day')
        # Set the y axis label of the current axis.
        plt.ylabel('y - number of instances')
        # Set a title of the current axes.
        plt.title(title)
        plt.show()


        # for list in list_of_elements:
        #     matrix = np.column_stack(matrix,list)



        return
def negihberhhod_overlap(G,number_of_edges):
    overlap_dict = dict()
    for edge in G.edges():
        common_neighbors = list(nx.common_neighbors(G,edge[0],edge[1]))
        all_neighbors = G.degree(edge[0]) + G.degree(edge[1])
        overlap_dict[edge] = len(common_neighbors) / all_neighbors
    overlap_dict = {k: v for k, v in sorted(overlap_dict.items(), key=lambda item: item[1])}
    number = 0
    for key in overlap_dict.keys():
        G.remove_edge(key[0],key[1])
        number += 1
        if number == number_of_edges:
            break
    print(len(G.edges()))
    return G
def removed_highest_degree(G,number_of_edges):
    node_degree_dict = dict()
    for node in G.nodes():
        node_degree_dict[node] = G.degree(node)
    node_degree_dict = dict(sorted(node_degree_dict.items(), key=lambda item: item[1], reverse=True))
    number = 0
    for key in node_degree_dict.keys():
        number += G.degree(key)
        G.remove_node(key)
        if number >= number_of_edges:
            break
    print(len(G.edges()))
    return G
def calculate_clustering_coefficient(G, number_of_edges):
        nodes_list = list(G.nodes())
        dict_cluster = dict()
        for node in nodes_list:
            number_triplrts = 0
            node_negihbors_list = list(G.neighbors(node))
            if (len(node_negihbors_list) < 2):
                dict_cluster[node] = 0
                continue
            else:
                combinations = comb(len(node_negihbors_list), 2)
            for node2 in node_negihbors_list:
                node2_negihbors_list = G.neighbors(node2)
                for node3 in node2_negihbors_list:
                    if node3 in node_negihbors_list:
                        number_triplrts = number_triplrts + 1
                        continue
            dict_cluster[node] = (number_triplrts / combinations) / 2
        dict_cluster = dict(sorted(dict_cluster.items(), key=lambda item: item[1], reverse=False))
        number = 0
        for key in dict_cluster.keys():
            number += G.degree(key)
            G.remove_node(key)
            if number >= number_of_edges:
                break
        print(dict_cluster)
        print(len(G.edges()))
        return G
