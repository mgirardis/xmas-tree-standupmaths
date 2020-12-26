import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def main():
    #numframes = 100
    numpoints = 10
    period = 15
    #color_data = np.random.randint(15,size=(numframes, numpoints))
    x, y, z = np.random.random((3, numpoints))
    color_arr = plt.get_cmap('viridis')(np.linspace(0,1,period))
    #print(color_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    scat = ax.scatter(x, y, z, c=np.zeros(numpoints), vmin=0, vmax=1, s=100, cmap=plt.get_cmap('viridis'))
    #scat = ax.scatter(x, y, z, c=color_arr[np.zeros(numpoints).astype(int)], vmin=0, vmax=1, s=100, cmap=plt.get_cmap('viridis'))
    #print(color_arr[np.zeros(numpoints).astype(int)])

    #ani = animation.FuncAnimation(fig, update_plot, fargs=(color_data, scat), blit=True)
    ani = animation.FuncAnimation(fig, update_plot, fargs=(scat,numpoints,period), blit=True)
    plt.show()

#def update_plot(i, data, scat):
#    scat.set_array(data[i])
def update_plot(i, scat, numpoints,period):
    #print((i%period)*np.ones(numpoints).astype(int))
    #scat.set_array((i%period)*np.ones(numpoints).astype(int))
    #print(((i%period)/period)*np.ones(numpoints))
    scat.set_array(((i%period)/period)*np.ones(numpoints))
    return scat,

main()