This program is a contribution to the twinkly xmas tree lights of the standupmaths youtube channel, related to the video https://youtu.be/TvlpIojusBE

Each LED is located in the position given in the file coords.txt (generated by standupmaths).
Each LED light follows the equation of the electric potential in the membrane of a formal neuron (described in http://dx.doi.org/10.1016/j.jneumeth.2013.07.014 and in https://doi.org/10.1371/journal.pone.0174621.g002 ).
Each neuron (i.e., LED light) receives input from its nearest neighbor neurons (i.e., LED lights) through a model of chemical synapses.

The network works collectively to generate (possibly chaotic) flashing patterns, that might include spiral waves (which would be guaranteed to happen if the tree was a 2d lattice), synchronization, etc.

Two example collective behaviors are given by two predetermined parameter sets, given by the input parameters 'spiral' and 'sync'.

Below, you can find the description of what each program here does.

# view_xmaslights_activity.py

The actual simulation of what the xmas tree will look like, visualized in a matplotlib animation.

Running this file like

```
python view_xmaslights_activity.py -set spiral
```

generates the collective flashing pattern that displays spiral waves (although it's not easy to spot them).

<a href="output_xmastree_activity/xmas_tree_spiral.mp4">Video example.</a>

```
python view_xmaslights_activity.py -set sync
```

generates synchronized bursts of activity intertwined with random firing of LEDs.

<a href="output_xmastree_activity/xmas_tree_sync.mp4">Video example.</a>


# view_tree.m

Simple visualization in MATLAB of the tree coordinates captured by standupmaths, and of the geometric cone generated by the data parameters.

<img src="example_xmastree_network/tree_cone.png" width="200" alt="Cone of the tree" />

# view_tree_network.py

Builds and shows the network of the xmas lights. The network can be built in 3 different ways:

- a lattice-like network (nearest neighbors in all 6 spatial directions)

<img src="example_xmastree_network/lattice.png" width="200" alt="Lattice network example" />

- a proximity network (a given light receives input from all the other lights within a certain radius of itself)

<img src="example_xmastree_network/radius.png" width="200" alt="Proximity network example" />

- a surface network (only the lights outside of the geometric cone generated by the data parameters are connected to its 4 nearest neighbors, in polar coordinates)

<img src="example_xmastree_network/surface.png" width="200" alt="Surface network example" />

# xmastree_network.py

The file that (hopefuly) can run on the standupmaths xmas tree. It should produce the same output as the one displayed by the simulation in view_xmaslights_activity.py
