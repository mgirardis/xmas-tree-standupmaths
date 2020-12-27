import numpy
import networkx
import xmaslights_network as xmt

r_LEDs = numpy.asarray(xmt.load_coords(),dtype=float)

print('Calculating cubic-like lattice network ...')

R_connection = 0.0
conic_surface_only = False
LED_neigh = xmt.generate_list_of_neighbors(r_LEDs,R_connection,on_conic_surface_only=conic_surface_only)
LED_edges = [ (i,n) for i,nl in enumerate(LED_neigh) for n in nl ]

# networkx does not plot a 3d graph :(
#G = networkx.Graph()
#G.add_edges_from(LED_edges)
#networkx.draw(G,pos=LED_pos_nx)

fh = xmt.plt.figure(figsize=(10,10))
ax = fh.add_subplot(111,projection='3d')
xmt.plot_network_3d(r_LEDs,LED_edges,node_args=dict(color='r',s=30),edge_args=dict(alpha=0.2),ax=ax)
ax.set_title('Cubic-like lattice')
xmt.plt.show()


print('Calculating connection radius network ...')

R_connection = xmt.mean_distance(r_LEDs) / 4.0
conic_surface_only = False
LED_neigh = xmt.generate_list_of_neighbors(r_LEDs,R_connection,on_conic_surface_only=conic_surface_only)
LED_edges = [ (i,n) for i,nl in enumerate(LED_neigh) for n in nl ]

fh = xmt.plt.figure(figsize=(10,10))
ax = fh.add_subplot(111,projection='3d')
xmt.plot_network_3d(r_LEDs,LED_edges,node_args=dict(color='r',s=30),edge_args=dict(alpha=0.2),ax=ax)
ax.set_title('Connection radius R={:g}'.format(R_connection))
xmt.plt.show()


print('Calculating surface network ...')

R_connection = 0.0
conic_surface_only = True
LED_neigh = xmt.generate_list_of_neighbors(r_LEDs,R_connection,on_conic_surface_only=conic_surface_only)
LED_edges = [ (i,n) for i,nl in enumerate(LED_neigh) for n in nl ]

fh = xmt.plt.figure(figsize=(10,10))
ax = fh.add_subplot(111,projection='3d')
xmt.plot_network_3d(r_LEDs,LED_edges,node_args=dict(color='r',s=30),edge_args=dict(alpha=0.5),ax=ax)
ax.set_title('Surface network')
xmt.plt.show()