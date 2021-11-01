import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('seaborn')

def set_data(ln, p1, p2):
    ln.set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

# get view angle
az = np.load('out_files/cam_az.npy', allow_pickle=True)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(10, az)
ax.set_xlim(-0.75,0.75)
ax.set_ylim(-0.75,0.75)
ax.set_zlim(0,1.5)
ax.set_axis_off()

colors = ['k','r','r','b','b','k','k','k','k','k',
          'r','r','b','b']

lns = []
# setup line segments
for color in colors:
    lns.append(ax.plot([0,0], [0,0], [0,0], color=color)[0])#, linewidth=12)[0])

plt.show(block=False)
plt.pause(0.01)
plt.tight_layout()

bg = fig.canvas.copy_from_bbox(fig.bbox)
for ln in lns: 
    ax.draw_artist(ln)
fig.canvas.blit(fig.bbox)

angles_save = np.zeros(9)
# read and plot joints (realtime with _.py)
while True:
    # read joints, 17 total points in a person.
    # p is short for person, [joint #, coordinate]
    try:
        p = np.load('out_files/joints.npy', allow_pickle=True)[0]
        floor = min(p[3,2], p[6,2]) # smallest val of either foot
        for i in range(p.shape[0]):
            p[i,2] -= floor # normalize z coordinate
    except:
        print('failed load')

    # plot lines between joints to recontruct stick figure
    fig.canvas.restore_region(bg)

    # hips
    set_data(lns[0], p[1], p[4])

    # right leg 
    set_data(lns[1], p[1], p[2])
    set_data(lns[2], p[2], p[3])

    # left leg 
    set_data(lns[3], p[4], p[5])
    set_data(lns[4], p[5], p[6])

    # spine + head
    set_data(lns[5], p[0], p[7])
    set_data(lns[6], p[7], p[8])
    set_data(lns[7], p[8], p[9])
    set_data(lns[8], p[9], p[10])
    
    # shoulder
    set_data(lns[9], p[11], p[14])

    # right arm 
    set_data(lns[10], p[14], p[15])
    set_data(lns[11], p[15], p[16])

    # left arm 
    set_data(lns[12], p[11], p[12])
    set_data(lns[13], p[12], p[13])

    for ln in lns: 
        ax.draw_artist(ln)

    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()




