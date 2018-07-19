import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import ipm,scatter

mpl.rcParams['font.size'] = 12.
mpl.rcParams['mathtext.default'] = 'sf'

points = 300 #Integration precision
shots = 100

ko = 12398
xstd,ystd   = 2000, 2000
beamx,beamy = np.random.normal(0., xstd, shots),np.random.normal(0., ystd, shots)

r = 1e28*np.vstack([ipm.ipm_readings(ko, x, y, points=points) for x,y in zip(beamx,beamy)])
ipmx = (r[:,1] - r[:,3]) / (r[:,1] + r[:,3])
ipmy = (r[:,0] - r[:,2]) / (r[:,0] + r[:,2])

f = plt.figure(figsize=(8, 3))
ax = f.add_subplot(121)
x,y = beamx,ipmx
plt.scatter(beamx, ipmx, facecolors='none', edgecolors='k')
plt.xlabel(r'$Beam\ X\ (\mu m)$')
plt.ylabel(r'$IPM_X\ \left( \frac{IPM_R - IPM_L}{IPM_R + IPM_L}\right)$', fontsize=16)
plt.ylim(y.min() - 0.1*(y.max() - y.min()) ,y.max() + 0.1*(y.max() - y.min()))
plt.xlim(-5000, 5000)
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')

ax = f.add_subplot(122)
x,y = beamy,ipmy
plt.scatter(beamy, ipmy, facecolors='none', edgecolors='k')
plt.xlabel(r'$Beam\ Y\ (\mu m)$')
plt.ylabel(r'$IPM_Y\ \left( \frac{IPM_T - IPM_B}{IPM_T + IPM_B} \right)$', fontsize=16, labelpad=14.)
plt.ylim(y.min() - 0.1*(y.max() - y.min()) ,y.max() + 0.1*(y.max() - y.min()))
plt.xlim(-5000, 5000)
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')

plt.tight_layout()
#plt.show()

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../renders/ipm_edges.png')
plt.savefig(filename, fmt='png', dpi=600)
