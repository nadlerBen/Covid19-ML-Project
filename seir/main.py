import math
import numpy as np
import pandas as pd
import networkx as nx
import random
from scipy.special import comb
import matplotlib.pyplot as plt
import functions as fu
days = 75
status_list = ['S','E','I','Is','Ic','R']
p_c = {'Young':0.005,'Adult':0.01,'MidAge':0.3,'Elder':0.6}
def main():
    data = fu.ModelData('graph_edges.csv')
    users_data = pd.read_csv('nodes.csv')
    question = 0
    days = 75
    dict_of_dicts_exposed = dict()
    dict_of_dicts_infected = dict()
    dict_of_dicts_subclinical = dict()
    dict_of_dicts_hospitalized = dict()
    dict_of_dicts_recovered = dict()
    x_list = list()
    while question <88:
        x_list = []
        question += 1
        Graph,exposed,infectious,sub_clinical,hosiptalized,recovered = fu.create_undirected_graph(data,users_data,question)
        ################################################# over lap mitigation strategy ####################################
        # Graph = fu.negihberhhod_overlap(Graph,25234)
        # Graph = fu.removed_highest_degree(Graph,25234)
        # Graph = fu.calculate_clustering_coefficient(Graph,25234)
        ################################################# over lap mitigation strategy ####################################
        daily_hospitelaized = dict()
        daily_exposed = dict()
        daily_infected = dict()
        daily_subclinical = dict()
        daily_recovered = dict()
        for i in range(days):
            x_list.append(i)
            R0, P_relation = fu.daily_update()
            exposed, infectious, sub_clinical, hosiptalized, recovered = fu.Stochastic_update(Graph,exposed, infectious, sub_clinical, hosiptalized,recovered,P_relation,question)
            exposed, infectious, sub_clinical, hosiptalized, recovered = fu.remove_from_dict(exposed, infectious, sub_clinical, hosiptalized, recovered)
            ############# update summery ##################
            daily_exposed[i] = len(exposed)
            daily_infected[i] = len(infectious)
            daily_subclinical[i] = len(sub_clinical)
            daily_hospitelaized[i] = len(hosiptalized)
            daily_recovered[i] = len(recovered)
        # fu.plot_graph(daily_exposed.keys(),daily_exposed.values(),daily_infected.values(),daily_subclinical.values(),daily_hospitelaized.values())
        dict_of_dicts_exposed[question] = daily_exposed
        dict_of_dicts_infected[question] = daily_infected
        dict_of_dicts_subclinical[question] = daily_subclinical
        dict_of_dicts_hospitalized[question] = daily_hospitelaized
        dict_of_dicts_recovered[question] = daily_recovered
    fu.plot_confidance_graph(days,dict_of_dicts_exposed,x_list,'b',"exposed confidance interval")
    fu.plot_confidance_graph(days,dict_of_dicts_infected,x_list,'y',"infected confidance interval")
    fu.plot_confidance_graph(days,dict_of_dicts_subclinical,x_list,'m',"subclinical confidance interval")
    fu.plot_confidance_graph(days,dict_of_dicts_hospitalized,x_list,'r',"hospitalized confidance interval")
    fu.plot_confidance_graph(days,dict_of_dicts_recovered,x_list,'g',"recovered confidance interval")

    return
if __name__ == '__main__':
    main()