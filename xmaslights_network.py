# -*- coding: utf-8 -*-

import re
import os
import argparse
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():

    r = numpy.asarray(load_coords(),dtype=float)

    # just visualizing the tree
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(r[:,0],r[:,1],r[:,2],marker='o')
    plt.show()

    anim_dt = 0.01 # animation interval in seconds

    neigh_c = generate_network(r) # cubic neighbors
    neigh_R = generate_network(r, mean_distance(r)/6)

    return
    fh = plt.figure(1)
    pdata = plt.imshow(img[0])
    ani = animation.FuncAnimation(fh, animate, len(img), fargs=(img,plt,L), interval=int(anim_dt*1000), blit=True)
    plt.show()

def network_time_step(V,S,parNeuron,parSyn):
    return

def synapse_map(i,S,par,Vpre):
    # par[0] -> J, par[1] -> noise amplitude, par[2] -> 1/tau_f, par[3] -> 1/tau_g
    thetaJ =  par[0] + (par[1] * numpy.random.random()) if Vpre > 0.0 else 0.0
    S[i,0] = (1.0 - par[2]) * S[i,0] + S[i,1]
    S[i,1] = (1.0 - par[3]) * S[i,1] + thetaJ
    return S

def logistic_func(u):
    return u / (1 + (u if u > 0.0 else -u)) # u/(1+|u|)
def neuron_map_log(i,V,par,S):
    # par[0] -> K, par[1] -> 1/T, par[2] -> d, par[3] -> l, par[4] -> xR
    Vprev = V[i,0]
    V[i,0] = logistic_func((V[i,0] - par[0] * V[i,1] + V[i,2] + numpy.sum(S))*par[1])
    V[i,1] = Vprev
    V[i,2] = (1.0 - par[2]) * V[i,2] - par[3] * (Vprev - par[4])
    return V

def neuron_map_tanh(i,V,par,S):
    # par[0] -> K, par[1] -> 1/T, par[2] -> d, par[3] -> l, par[4] -> xR
    Vprev = V[i,0]
    V[i,0] = numpy.tanh((V[i,0] - par[0] * V[i,1] + V[i,2] + numpy.sum(S))*par[1])
    V[i,1] = Vprev
    V[i,2] = (1.0 - par[2]) * V[i,2] - par[3] * (Vprev - par[4])
    return V

def mean_distance(r):
    n = r.shape[0]
    dsum = 0.0
    N = 0
    i = 0
    while i < n:
        j = i+1
        while j < n:
            N += 1
            dsum += numpy.linalg.norm(r[i,:]-r[j,:])
            j += 1
        i += 1
    return dsum / float(N)

def generate_network(r,R=0.0):
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

def animate(t,img,plt,L):
    pdata = plt.imshow(img[t])
    ax = plt.gca()
    ax.set_xticks(numpy.arange(L[1]))
    ax.set_yticks(numpy.arange(L[0]))
    #print('t = %d'%t)
    #print(img[t])
    return pdata,


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