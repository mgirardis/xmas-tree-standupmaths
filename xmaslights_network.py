# -*- coding: utf-8 -*-

import re
import os
import argparse
import numpy
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'D:\\Dropbox\\p\\programas\\bin\\ffmpeg\\bin\\ffmpeg.exe'
import matplotlib.animation as animation

def main():

    #global r_LEDs
    r_LEDs = numpy.asarray(load_coords(),dtype=float)

    # just visualizing the tree
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')
    #ax.scatter(r_LEDs[:,0],r_LEDs[:,1],r_LEDs[:,2],marker='o')
    #plt.show()

    # setting animation parameters
    color_arr_10 = numpy.asarray([[0.267004, 0.004874, 0.329415, 1.      ],
                                  [0.281412, 0.155834, 0.469201, 1.      ],
                                  [0.244972, 0.287675, 0.53726 , 1.      ],
                                  [0.190631, 0.407061, 0.556089, 1.      ],
                                  [0.147607, 0.511733, 0.557049, 1.      ],
                                  [0.119699, 0.61849 , 0.536347, 1.      ],
                                  [0.20803 , 0.718701, 0.472873, 1.      ],
                                  [0.430983, 0.808473, 0.346476, 1.      ],
                                  [0.709898, 0.868751, 0.169257, 1.      ],
                                  [0.993248, 0.906157, 0.143936, 1.      ]])

    color_arr_15 = numpy.asarray([[0.267004, 0.004874, 0.329415, 1.      ],
                                  [0.28291 , 0.105393, 0.426902, 1.      ],
                                  [0.275191, 0.194905, 0.496005, 1.      ],
                                  [0.248629, 0.278775, 0.534556, 1.      ],
                                  [0.212395, 0.359683, 0.55171 , 1.      ],
                                  [0.180629, 0.429975, 0.557282, 1.      ],
                                  [0.153364, 0.497   , 0.557724, 1.      ],
                                  [0.127568, 0.566949, 0.550556, 1.      ],
                                  [0.122312, 0.633153, 0.530398, 1.      ],
                                  [0.175707, 0.6979  , 0.491033, 1.      ],
                                  [0.288921, 0.758394, 0.428426, 1.      ],
                                  [0.449368, 0.813768, 0.335384, 1.      ],
                                  [0.626579, 0.854645, 0.223353, 1.      ],
                                  [0.814576, 0.883393, 0.110347, 1.      ],
                                  [0.993248, 0.906157, 0.143936, 1.      ]])

    color_arr_20 = numpy.asarray([[0.267004, 0.004874, 0.329415, 1.      ],
                                  [0.280894, 0.078907, 0.402329, 1.      ],
                                  [0.28229 , 0.145912, 0.46151 , 1.      ],
                                  [0.270595, 0.214069, 0.507052, 1.      ],
                                  [0.250425, 0.27429 , 0.533103, 1.      ],
                                  [0.223925, 0.334994, 0.548053, 1.      ],
                                  [0.19943 , 0.387607, 0.554642, 1.      ],
                                  [0.175841, 0.44129 , 0.557685, 1.      ],
                                  [0.15627 , 0.489624, 0.557936, 1.      ],
                                  [0.136408, 0.541173, 0.554483, 1.      ],
                                  [0.121831, 0.589055, 0.545623, 1.      ],
                                  [0.12478 , 0.640461, 0.527068, 1.      ],
                                  [0.162016, 0.687316, 0.499129, 1.      ],
                                  [0.239374, 0.735588, 0.455688, 1.      ],
                                  [0.335885, 0.777018, 0.402049, 1.      ],
                                  [0.458674, 0.816363, 0.329727, 1.      ],
                                  [0.585678, 0.846661, 0.249897, 1.      ],
                                  [0.730889, 0.871916, 0.156029, 1.      ],
                                  [0.866013, 0.889868, 0.095953, 1.      ],
                                  [0.993248, 0.906157, 0.143936, 1.      ]])
    #global color_arr
    anim_dt = 0.02 # animation interval in seconds
    color_arr = color_arr_15
    color_map = plt.get_cmap('viridis')

    # setting external stimulus parameters
    r_Poisson = 0.2 # rate of Poisson process
    P_Poisson = 1.0-numpy.exp(-r_Poisson) # probability of firing is constant

    # setting neuron parameters
    parNeuron_tanh = [ 0.6, 1.0/0.35, 0.001, 0.008, -0.7, 0.1 ] # par[0] -> K, par[1] -> 1/T, par[2] -> d, par[3] -> l, par[4] -> xR, par[5] -> Iext
    neuron_map_iter = neuron_map_tanh
    parNeuron = parNeuron_tanh

    # setting synapse parameters
    parSynapse = [-0.2,0.0,1.0/2.0,1.0/2.0] # par[0] -> J, par[1] -> noise amplitude, par[2] -> 1/tau_f, par[3] -> 1/tau_g
    R_connection = 0.0 # if 0, generates a cubic lattice; if > 0, then connects all pixels that are within radius R of each other

    global V,S
    V,S,input_list,presyn_neuron_list = build_network(r_LEDs,R=R_connection)
    V = set_initial_condition(V,neuron_map_iter,parNeuron_tanh)

    fh = plt.figure(1)
    ax = fh.add_subplot(111,projection='3d')
    splot = ax.scatter(r_LEDs[:,0],r_LEDs[:,1],r_LEDs[:,2],c=memb_potential_to_01(V),vmin=0,vmax=1,cmap=color_map)
    ani = animation.FuncAnimation(fh, animate, fargs=(splot,neuron_map_iter,parNeuron,input_list,presyn_neuron_list,parSynapse,P_Poisson), interval=int(anim_dt*1000), blit=True, repeat=True, save_count=100)
    plt.show()
    ani.save('xmas_tree_sim.mp4',fps=15)
    #L = [20,25]
    #fh = plt.figure(1)
    #pdata = plt.imshow(((V[:,0]+1.0)/2.0).reshape(L),vmin=0.0,vmax=1.0)
    #ani = animation.FuncAnimation(fh, animate, fargs=(plt,L,neuron_map_iter,parNeuron,input_list,presyn_neuron_list,parSynapse,P_Poisson), interval=int(anim_dt*1000), blit=True)
    #plt.show()

def memb_potential_to_01(V):
    # V -> dynamic variable
    return ((V[:,0]+1.0)/2.0)**4

def memb_potential_to_coloridx(V,n_colors):
    # V -> dynamic variable
    # n_colors -> total number of colors
    return numpy.floor(n_colors*memb_potential_to_01(V)).astype(int)

def set_initial_condition(V,neuron_map_iter,parNeuron):
    V0 = get_neuron_resting_state(neuron_map_iter,parNeuron)
    i = 0
    while i < V.shape[0]:
        V[i,:] = V0.copy()
        i+=1
    return V

def build_network(r_nodes,R=0.0):
# r_nodes vector of coordinates of each pixel
# R connection radius
    # if R is zero, generates an attempt of a cubic-like lattice, otherwise connects all pixels within a radius R of each other
    neigh = generate_list_of_neighbors(r_nodes,R)
    # creates the interaction lists between dynamic variables
    input_list,presyn_neuron_list = create_input_lists(neigh)
    # creates dynamic variables
    N = len(neigh) # number of neurons (or pixels)
    Nsyn = len(presyn_neuron_list)
    V = numpy.zeros((N,3)) # membrane potential (dynamic variables) of each neuron (pixel)
    S = numpy.zeros((Nsyn,2)) # synaptic current input generated by each pixel towards each of its postsynaptic pixels
    return V,S,input_list,presyn_neuron_list

def create_input_lists(neigh):
    # given a list of neighbors, where neigh[i] is a list of inputs to node i
    # generate the list of inputs to be used in the simulation
    presyn_neuron_list = [n for sublist in neigh for n in sublist]
    cs = numpy.insert(numpy.cumsum([ n.size for n in neigh ]),0,0)
    input_list = [ numpy.arange(a,b) for a,b in zip(cs[:-1],cs[1:]) ]
    return input_list,presyn_neuron_list

def network_time_step(neuron_map_iter,V,parNeuron,input_list,S,presyn_neuron_list,parSyn,P_poisson):
# neuron_map_iter -> function that iterates the neuron (either neuron_map_log or neuron_map_tanh)
# V -> numpy.ndarray with shape (N,3), where N is the number of neurons (pixels), containing the membrane potential of neurons
# parNeuron -> list of six parameters that is passed to the neuron_map_iter
# input_list -> list of neighbors of each pixel, such that element i: list of neighbors (rows of S) that send input to pixel i
# S -> numpy.ndarray with shape (Nsyn,2), where Nsyn is the total number of synapses (connections between pixels), containg the synaptic current of each connection
# presyn_neuron_list -> list of presynaptic neurons (i.e. rows of V) that generates each synapse, such that pixel given by presyn_neuron_list[i] generates synapse S[i,:]
# parSyn -> list of 4 parameters that is passed to synapse_map function
# P_poisson -> probability of generating a random activation of a random pixel
    if numpy.random.random() < P_poisson:
        k = numpy.random.randint(V.shape[0]) # selects a random pixel to stimulate
        Iext = parNeuron[5] # external kick
    else:
        k = 0
        Iext = 0.0
    i = 0
    while i < S.shape[0]:
        S = synapse_map(i,S,parSyn,V[presyn_neuron_list[i],0]) # evolve the synaptic equations
        i+=1
    i = 0
    while i < k:
        V = neuron_map_iter(i,V,parNeuron,S[input_list[i],0],0.0) # evolve the pixel (neuron) equations
        i+=1
    V = neuron_map_iter(i,V,parNeuron,S[input_list[i],0],Iext) # evolve the pixel (neuron) equations
    i+=1
    while i < V.shape[0]:
        V = neuron_map_iter(i,V,parNeuron,S[input_list[i],0],0.0) # evolve the pixel (neuron) equations
        i+=1
    return V,S

def synapse_map(i,S,par,Vpre):
    # par[0] -> J, par[1] -> noise amplitude, par[2] -> 1/tau_f, par[3] -> 1/tau_g
    thetaJ =  par[0] + (par[1] * numpy.random.random()) if Vpre > 0.0 else 0.0
    S[i,0] = (1.0 - par[2]) * S[i,0] + S[i,1]
    S[i,1] = (1.0 - par[3]) * S[i,1] + thetaJ
    return S

def get_neuron_resting_state(neuron_map_iter,par,T=20000):
    V = -0.9*numpy.ones((1,3))
    t = 0
    while t<T:
        V = neuron_map_iter(0,V,par,[],0.0)
        t+=1
    return V

def logistic_func(u):
    return u / (1 + (u if u > 0.0 else -u)) # u/(1+|u|)
def neuron_map_log(i,V,par,S,Iext):
    # par[0] -> K, par[1] -> 1/T, par[2] -> d, par[3] -> l, par[4] -> xR
    Vprev = V[i,0]
    V[i,0] = logistic_func((V[i,0] - par[0] * V[i,1] + V[i,2] + numpy.sum(S)+Iext)*par[1])
    V[i,1] = Vprev
    V[i,2] = (1.0 - par[2]) * V[i,2] - par[3] * (Vprev - par[4])
    return V

def neuron_map_tanh(i,V,par,S,Iext):
    # par[0] -> K, par[1] -> 1/T, par[2] -> d, par[3] -> l, par[4] -> xR
    Vprev = V[i,0]
    V[i,0] = numpy.tanh((V[i,0] - par[0] * V[i,1] + V[i,2] + numpy.sum(S)+Iext)*par[1])
    V[i,1] = Vprev
    V[i,2] = (1.0 - par[2]) * V[i,2] - par[3] * (Vprev - par[4])
    return V

def mean_distance(p):
    n = p.shape[0]
    dsum = 0.0
    N = 0
    i = 0
    while i < n:
        j = i+1
        while j < n:
            N += 1
            dsum += numpy.linalg.norm(p[i,:]-p[j,:])
            j += 1
        i += 1
    return dsum / float(N)

def generate_list_of_neighbors(r,R=0.0):
    # generates a network of "pixels"
    # each pixel in position r[i,:] identifies its 6 closest neighbors and should receive a connection from it
    # if R is given, includes all pixels within a radius R of r[i,:] as a neighbor
    # the 6 neighbors are chosen such that each one is positioned to the left, right, top, bottom, front or back of each pixel (i.e., a simple attempt of a cubic lattice)
    #
    # r -> position vector (each line is the position of each pixel)
    # R -> neighborhood ball around each pixel
    #
    # returns:
    #    list of neighbors
    #     neigh[i] -> list of 6 "pixels" closest to i
    def get_first_val_not_in_list(v,l):  # auxiliary function
        # returns first value in v that is not in l
        if v.size == 0:
            return None
        n = len(v)
        i = 0
        while i < n:
            if not (v[i] in l):
                return v[i]
            i+=1
    #get_existing_neigh = lambda n: [ m for m in n if not (m is None) ] # another auxiliary function
    neigh = []
    for r0 in r:
        if (R>0.0): # a neighborhood radius is given
            neigh.append(numpy.nonzero(numpy.linalg.norm(r-r0,axis=1)<R)[0])
        else: # a radius is not given, hence returns a crystalline-like cubic-like structure :P
            pixel_list_sorted = numpy.argsort(numpy.linalg.norm(r-r0,axis=1)) # sorted by Euler distance to r0
            rs = r[pixel_list_sorted,:] # list of positions from the closest to the farthest one to r0
            local_neigh_list = [] # local neighbor list
            x1_neigh = get_first_val_not_in_list(numpy.nonzero(rs[:,0]<r0[0])[0],local_neigh_list) # gets first neighbor to the left that is not added yet
            if x1_neigh:
                local_neigh_list.append(x1_neigh)
            x2_neigh = get_first_val_not_in_list(numpy.nonzero(rs[:,0]>r0[0])[0],local_neigh_list) # gets first neighbor to the right that is not added yet
            if x2_neigh:
                local_neigh_list.append(x2_neigh)
            y1_neigh = get_first_val_not_in_list(numpy.nonzero(rs[:,1]<r0[1])[0],local_neigh_list) # gets first neighbor to the back that is not added yet
            if y1_neigh:
                local_neigh_list.append(y1_neigh)
            y2_neigh = get_first_val_not_in_list(numpy.nonzero(rs[:,1]>r0[1])[0],local_neigh_list) # gets first neighbor to the front that is not added yet
            if y2_neigh:
                local_neigh_list.append(y2_neigh)
            z1_neigh = get_first_val_not_in_list(numpy.nonzero(rs[:,2]<r0[2])[0],local_neigh_list) # gets first neighbor to the top that is not added yet
            if z1_neigh:
                local_neigh_list.append(z1_neigh)
            z2_neigh = get_first_val_not_in_list(numpy.nonzero(rs[:,2]>r0[2])[0],local_neigh_list) # gets first neighbor to the bottom that is not added yet
            if z2_neigh:
                local_neigh_list.append(z2_neigh)
            neigh.append(pixel_list_sorted[local_neigh_list]) # adds neighbors
    return neigh

#def animate(t,ax,L,neuron_map_iter,parNeuron,input_list,presyn_neuron_list,parSyn,P_poisson):
def animate(t,splot,neuron_map_iter,parNeuron,input_list,presyn_neuron_list,parSyn,P_poisson):
    global V,S
    V,S = network_time_step(neuron_map_iter,V,parNeuron,input_list,S,presyn_neuron_list,parSyn,P_poisson)
    splot.set_array(memb_potential_to_01(V))
    return splot,


def load_coords(coordfilename = "xmastree2020/coords.txt"):
    fin = open(coordfilename,'r')
    coords_raw = fin.readlines()
    coords_bits = [i.split(",") for i in coords_raw]
    coords = []
    for slab in coords_bits:
        new_coord = []
        for i in slab:
            new_coord.append(int(re.sub(r'[^-\d]','', i)))
        coords.append(new_coord)
    return coords

if __name__ == '__main__':
    main()