# -*- coding: utf-8 -*-

import sys
import re
import os
import argparse
import numpy
import matplotlib.pyplot as plt
#print(plt.rcParams['animation.ffmpeg_path']) # if it does not recognize the ffmpeg, force it to look in the right place
#plt.rcParams['animation.ffmpeg_path'] = 'D:\\Dropbox\\p\\programas\\bin\\ffmpeg\\bin\\ffmpeg.exe'
import matplotlib.animation as animation
from numpy.core.fromnumeric import mean

def main():

    parser = argparse.ArgumentParser(description='Simulates xmas tree with LED coordinates from standupmaths channel:\n\n       https://youtu.be/TvlpIojusBE')
    parser.add_argument('-save', nargs=1, required=False, metavar='NUM_OF_FRAMES', type=int, default=[0], help='if 0, doesnt save video; otherwise saves the amount of frames given by this parameter')
    parser.add_argument('-dt', nargs=1, required=False, metavar='SECONDS', type=float, default=[0.04], help='(seconds) animation interval')
    parser.add_argument('-R', nargs=1, required=False, metavar='RADIUS', type=float, default=[0.0], help='if > 0, connects every led within R distant from one another; otherwise generates a cubic-like lattice')
    parser.add_argument('-cmap', nargs=1, required=False, metavar='NAME', type=str, default=['viridis'], help='name of the colormap to use -- only the predefined by matplotlib are accepted')
    parser.add_argument('-set', nargs=1, required=False, metavar='CHOICE',type=str,default=['spiral'], choices=['spiral', 'sync'], help='spiral: generates spiral waves; sync: generates synchronized flashes')
    args = parser.parse_args()

    # loading the positions of each LED
    r_LEDs = numpy.asarray(load_coords(),dtype=float)

    # setting simulation params
    anim_dt = args.dt[0] # animation interval in seconds
    color_map = plt.get_cmap(args.cmap[0])
    save_video = args.save[0]
    sim_setting = args.set[0]
    R_connection = args.R[0] # if 0, generates a cubic lattice; if > 0, then connects all pixels that are within radius R of each other

    print('chosen settings: %s'%sim_setting)

    if sim_setting == 'spiral':
        # setting external stimulus parameters
        r_Poisson = 0.2 # rate of Poisson process
        # setting neuron parameters
        parNeuron_tanh = [ 0.6, 1.0/0.35, 0.001, 0.008, -0.7, 0.1 ] # par[0] -> K, par[1] -> 1/T, par[2] -> d, par[3] -> l, par[4] -> xR, par[5] -> Iext
        neuron_map_iter = neuron_map_tanh
        parNeuron = parNeuron_tanh
        V0 = None # uses default initial condition
        # setting synapse parameters
        #parSynapse = [-0.8,0.0,1.0/2.0,1.0/2.0] # par[0] -> J, par[1] -> noise amplitude, par[2] -> 1/tau_f, par[3] -> 1/tau_g
        parSynapse = [-0.2,0.0,1.0/2.0,1.0/2.0] # par[0] -> J, par[1] -> noise amplitude, par[2] -> 1/tau_f, par[3] -> 1/tau_g
        conic_surface_only = False
    elif sim_setting == 'sync':
        # setting external stimulus parameters
        r_Poisson = 0.0 # rate of Poisson process
        # setting neuron parameters
        parNeuron_tanh = [ 0.6, 1.0/0.35, 0.001, 0.001, -0.5, 0.1 ] # par[0] -> K, par[1] -> 1/T, par[2] -> d, par[3] -> l, par[4] -> xR, par[5] -> Iext
        neuron_map_iter = neuron_map_tanh
        parNeuron = parNeuron_tanh
        V0 = 'rest'
        # setting synapse parameters
        #parSynapse = [0.01,0.0,1.0/2.0,1.0/2.0] # par[0] -> J, par[1] -> noise amplitude, par[2] -> 1/tau_f, par[3] -> 1/tau_g
        parSynapse = [-0.2,0.05,1.0/5.0,1.0/5.0] # par[0] -> J, par[1] -> noise amplitude, par[2] -> 1/tau_f, par[3] -> 1/tau_g
        #R_connection = mean_distance(r_LEDs)/6.0
        conic_surface_only = True
    else:
        raise ValueError('unknown setting')

    global V,S
    P_Poisson = 1.0-numpy.exp(-r_Poisson) # probability of firing is constant
    V,S,input_list,presyn_neuron_list = build_network(r_LEDs,R=R_connection,conic_surface_only=conic_surface_only)
    V = set_initial_condition(V,neuron_map_iter,parNeuron_tanh,V0)

    fh = plt.figure(figsize=(10,10))
    ax = fh.add_subplot(111,projection='3d')
    ax = fix_aspect(ax,r_LEDs)
    splot = ax.scatter(r_LEDs[:,0],r_LEDs[:,1],r_LEDs[:,2],c=memb_potential_to_01(V),vmin=0,vmax=1,cmap=color_map,s=50,edgecolors=[0,0,0])
    ani = animation.FuncAnimation(fh, animate, fargs=(splot,neuron_map_iter,parNeuron,input_list,presyn_neuron_list,parSynapse,P_Poisson), interval=int(anim_dt*1000), blit=True, repeat=True, save_count=save_video)
    plt.show()
    if save_video > 0:
        fileName = 'xmas_tree_sim'
        try:
            print('... saving video: %s'%(fileName+'.mp4'))
            ani.save(fileName+'.mp4',fps=20)
        except:
            print('... saving video: %s'%(fileName+'.gif'))
            ani.save(fileName+'.gif',fps=15)


def memb_potential_to_01(V):
    # V -> dynamic variable
    return ((V[:,0]+1.0)*0.5)**4 # raising to 4 is just to emphasize bright colors

def memb_potential_to_coloridx(V,n_colors):
    # V -> dynamic variable
    # n_colors -> total number of colors
    return numpy.floor(n_colors*memb_potential_to_01(V)).astype(int)

def set_initial_condition(V,neuron_map_iter,parNeuron,V0_type=None):
    if type(V0_type) is type(None):
        V0 = get_neuron_resting_state(neuron_map_iter,parNeuron)
    else:
        if type(V0_type) is str:
            if V0_type == 'rest':
                V0 = get_neuron_resting_state(neuron_map_iter,parNeuron)
            elif V0_type == 'random':
                V = 2.0*numpy.random.random_sample(V.shape)-1.0
                return V
            else:
                raise ValueError('V0_type must be either an array or list with 3 elements or one of the following: rest, random')
        elif type(V0_type) is list:
            V0 = numpy.asarray(V0_type)
        else:
            if type(V0_type) is numpy.ndarray:
                V0 = V0_type
            else:
                raise ValueError('V0_type must be either an array or list with 3 elements or one of the following: rest, random')
    i = 0
    while i < V.shape[0]:
        V[i,:] = V0.copy()
        i+=1
    return V

def build_network(r_nodes,R=0.0,conic_surface_only=False):
# r_nodes vector of coordinates of each pixel
# R connection radius
    # if R is zero, generates an attempt of a cubic-like lattice, otherwise connects all pixels within a radius R of each other
    neigh = generate_list_of_neighbors(r_nodes,R,on_conic_surface_only=conic_surface_only)
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

def generate_list_of_neighbors(r,R=0.0,on_conic_surface_only=False):
    # generates a network of "pixels"
    # each pixel in position r[i,:] identifies its 6 closest neighbors and should receive a connection from it
    # if R is given, includes all pixels within a radius R of r[i,:] as a neighbor
    # the 6 neighbors are chosen such that each one is positioned to the left, right, top, bottom, front or back of each pixel (i.e., a simple attempt of a cubic lattice)
    #
    # r -> position vector (each line is the position of each pixel)
    # R -> neighborhood ball around each pixel
    # on_conic_surface_only -> if true, only links pixels that are on the conic shell of the tree
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
    if on_conic_surface_only:
        # only adds 4 neighbors (top, bottom, left, right) that are outside of the cone defined by the estimated tree cone parameters
        # cone equation (x**2 + y**2)/c**2 = (z-z0)**2
        z0 = numpy.max(r[:,2]) # cone height above the z=0 plane
        h = z0 + numpy.abs(numpy.min(r[:,2])) # cone total height
        base_r = (numpy.max(  (numpy.max(r[:,1]),numpy.max(r[:,0]))   ) + numpy.abs(numpy.min(  ( numpy.min(r[:,1]),numpy.min(r[:,0]) )  )))/2.0 # cone base radius
        c = base_r / h # cone opening radius (defined by wolfram https://mathworld.wolfram.com/Cone.html )
        #z_cone = lambda x,y,z0,c,s: z0+s*numpy.sqrt((x**2+y**2)/(c**2)) # s is the concavity of the cone: -1 turned down, +1 turned up
        cone_r_sqr = lambda z,z0,c: (c*(z-z0))**2
        outside_cone = (r[:,0]**2+r[:,1]**2) > cone_r_sqr(r[:,2],z0,c)
        pixel_list = numpy.nonzero(outside_cone)[0]
        r_out = r[outside_cone,:]

        neigh = [ numpy.array([],dtype=int) for i in range(r.shape[0]) ]
        for i,r0 in enumerate(r_out):
            # a radius is not given, hence returns a crystalline-like cubic-like structure :P
            pixel_list_sorted = numpy.argsort(numpy.linalg.norm(r_out-r0,axis=1)) # sorted by Euler distance to r0
            rs = r_out[pixel_list_sorted,:] # list of positions from the closest to the farthest one to r0
            local_neigh_list = [] # local neighbor list
            x1_neigh = get_first_val_not_in_list(numpy.nonzero( is_left_neigh(rs[:,:2],r0[:2]) )[0],local_neigh_list) # gets first neighbor to the left that is not added yet
            if x1_neigh:
                local_neigh_list.append(x1_neigh)
            x2_neigh = get_first_val_not_in_list(numpy.nonzero( numpy.logical_not(is_left_neigh(rs[:,:2],r0[:2])) )[0],local_neigh_list) # gets first neighbor to the right that is not added yet
            if x2_neigh:
                local_neigh_list.append(x2_neigh)
            z1_neigh = get_first_val_not_in_list(numpy.nonzero(rs[:,2]<r0[2])[0],local_neigh_list) # gets first neighbor to the top that is not added yet
            if z1_neigh:
                local_neigh_list.append(z1_neigh)
            z2_neigh = get_first_val_not_in_list(numpy.nonzero(rs[:,2]>r0[2])[0],local_neigh_list) # gets first neighbor to the bottom that is not added yet
            if z2_neigh:
                local_neigh_list.append(z2_neigh)
            neigh[pixel_list[i]] = pixel_list[pixel_list_sorted[local_neigh_list]] # adds neighbors
        return neigh
            
    neigh = []
    for r0 in r:
        if (R>0.0): # a neighborhood radius is given
            neigh.append(numpy.nonzero(numpy.linalg.norm(r-r0,axis=1)<R)[0])
        else:
            # a radius is not given, hence returns a crystalline-like cubic-like structure :P
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

def is_left_neigh(u,v):
    # u and v are two vectors on the x,y plane
    # u may be a list of vectors (one vector per row)
    return numpy.dot(u,[-v[1],v[0]])>0.0 # # the vector [-v[1],v[0]] is the 90-deg CCW rotated version of v

def myatan(x,y):
    return numpy.pi*(1.0-0.5*(1+numpy.sign(x))*(1-numpy.sign(y**2))-0.25*(2+numpy.sign(x))*numpy.sign(y))-numpy.sign(x*y)*numpy.arctan((numpy.abs(x)-numpy.abs(y))/(numpy.abs(x)+numpy.abs(y)))

def myacos(x):
    x[numpy.abs(x) > 1.0] = numpy.round(x[numpy.abs(x) > 1.0])
    return numpy.arccos(x)

def angle_uv(u,v,axis=None):
        return myacos(numpy.dot(u,v)/(numpy.linalg.norm(u,axis=axis)*numpy.linalg.norm(v)))

def get_color_matrix():
    return numpy.asarray([[0.267004, 0.004874, 0.329415, 1.      ],
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

def fix_aspect(ax,r):
    max_range = numpy.array([r[:,0].max()-r[:,0].min(), r[:,1].max()-r[:,1].min(), r[:,2].max()-r[:,2].min()]).max() / 2.0
    mid_x = (r[:,0].max()+r[:,0].min()) * 0.5
    mid_y = (r[:,1].max()+r[:,1].min()) * 0.5
    mid_z = (r[:,2].max()+r[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    return ax

def plot_network_3d(r,edge_list,node_args=None,edge_args=None,ax=None):
# r -> 3d position vector list: r[0] -> position of node 0, etc
# edge_list -> list of edges: [ (0,1), (0,4) ], means that node 0 is connected to node 1 and node 4, etc
# node_args -> args dict passed on to the node plotting function (matplotlib scatter)
# edge_args -> args dict passed on to the edge plotting function (matplotlib plot)
    if not ax:
        fh = plt.figure()
        ax = fh.add_subplot(111,projection='3d')
    ax = fix_aspect(ax,r)
    if not node_args:
        node_args = {}
    if not edge_args:
        edge_args = {}
    if 'c' not in edge_args.keys():
        edge_args['c'] = 'k'
    if 'alpha' not in edge_args.keys():
        edge_args['alpha'] = 0.5
    ax.scatter(r[:,0],r[:,1],r[:,2],**node_args)
    for e in edge_list:
        n1 = e[0]
        n2 = e[1]
        ax.plot([ r[n1,0],r[n2,0] ],[ r[n1,1],r[n2,1] ],[ r[n1,2],r[n2,2] ],**edge_args)
    return ax


#def animate(t,ax,L,neuron_map_iter,parNeuron,input_list,presyn_neuron_list,parSyn,P_poisson):
def animate(t,splot,neuron_map_iter,parNeuron,input_list,presyn_neuron_list,parSyn,P_poisson):
    global V,S
    V,S = network_time_step(neuron_map_iter,V,parNeuron,input_list,S,presyn_neuron_list,parSyn,P_poisson)
    #print(V[1,:])
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