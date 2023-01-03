import networkx as nx
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
from pandapower.estimation import estimate
import random
import matplotlib.pyplot as plt
import copy
import numpy as np
import re
from scipy.linalg import solve, null_space


def load_change(oval):
    lower_bound = oval * 0.8
    upper_bound = oval * 1.2
    return random.uniform(lower_bound, upper_bound)


def magnitude_change():
    return np.random.normal(loc=1, scale=0.02, size=1)


def gen_change(oval):
    return random.uniform(0, 100+oval)


def net_change(net):
    # change load information
    for i in range(len(net.load)):
        net.load.p_mw[i] = load_change(net.load.p_mw[i])
        net.load.q_mvar[i] = load_change(net.load.q_mvar[i])

    # change gen information
    for i in range(len(net.gen)):
        net.gen.p_mw[i] = gen_change(net.gen.p_mw[i])
        net.gen.vm_pu[i] = magnitude_change()

    return net

# net - power network
# SAMPLE_SIZE - the number of measurements (value between 0 and 744)
# NOISE - noise we add into each measurements


def gen_state_meas(net, SAMPLE_SIZE, NOISE):
    gen_index = random.randint(0, len(net.gen)-1)

    # change gen & load information
    try:
        net = net_change(net)
        pp.runpp(net, run_control=True)
    except:
        print("FAIL at power flow convergence")
        return None
    leader = net.gen.bus[gen_index]
    true_val = net.res_bus.p_mw[leader]

    net_s = copy.deepcopy(net)
    # random list of 173*4 + 13*4 = 692 + 52 = 744
    # [line p] [line q] [line p_to] [line q_to]
    # [trafo p] [trafo q] [trafo p_lv] [trafo q_lv]
    rand_lst = random.sample([i for i in range(744)], SAMPLE_SIZE)
    # print(rand_lst)
    ret = np.zeros(SAMPLE_SIZE)
    nodes = []

    # randomly select bus and line measurements
    for i in range(len(rand_lst)):
        pos = 0
        element = 0
        item = rand_lst[i]
        # trafo measurements
        if item < 52:
            element = int(item/13)
            pos = int(item % 13)
            if element == 0:
                oval = net.res_trafo.p_hv_mw[pos]
            elif element == 2:
                oval = net.res_trafo.q_hv_mvar[pos]
            elif element == 1:
                oval = net.res_trafo.p_lv_mw[pos]
            elif element == 3:
                oval = net.res_trafo.q_lv_mvar[pos]
            else:
                print("ERROR!", item, "should not fall into line measurements.")
            nodes.append((net.trafo.hv_bus[pos], net.trafo.lv_bus[pos]))
        # line measurements:
        else:
            element = int((item-52)/173)
            pos = int((item-52) % 173)
            if element == 0:
                oval = net.res_line.p_from_mw[pos]
            elif element == 2:
                oval = net.res_line.q_from_mvar[pos]
            elif element == 1:
                oval = net.res_line.p_to_mw[pos]
            elif element == 3:
                oval = net.res_line.q_to_mvar[pos]
            else:
                print("ERROR!", item, "should not fall into line measurements.")
            nodes.append((net.line.from_bus[pos], net.line.to_bus[pos]))
            pos = pos + 13
        # val = np.random.normal(loc=oval, scale=abs(oval*NOISE), size=1)
        val = oval
        ret[i] = val
    Y = np.concatenate(((np.array(net_s.res_bus.vm_pu)), np.array(net_s.res_bus.va_degree)),axis=0) # state vector
    # print(nodes)
    return ret, Y

net = pn.case118()
import pandapower.topology as top
mg = top.create_nxgraph(net)

# print(net)


import csv

# open the file in the write mode
f = open('./meas.csv', 'w')
f2 = open('./state.csv', 'w')

# create the csv writer
writer = csv.writer(f)
writer2 = csv.writer(f2)

i = 0
while i<10000:
    try:
        X, Y = gen_state_meas(net, 744, 0)
        # write a row to the csv file
        writer.writerow(X)
        writer2.writerow(Y)
        i+=1
    except:
        pass

# close the file
f.close()
f2.close()



# print(X.shape,Y.shape)
