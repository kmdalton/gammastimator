import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import ipm,scatter

mpl.rcParams['font.size'] = 12.
mpl.rcParams['mathtext.default'] = 'sf'

points = 300 #Integration precision
shots = 25

ko = 12398
xstd,ystd   = 100, 100
beamx,beamy = np.random.normal(0., xstd, shots),np.random.normal(0., ystd, shots)

r = 1e28*np.vstack([ipm.ipm_readings(ko, x, y, points=points) for x,y in zip(beamx,beamy)])
ipmx = (r[:,1] - r[:,3]) / (r[:,1] + r[:,3])
ipmy = (r[:,0] - r[:,2]) / (r[:,0] + r[:,2])

f = plt.figure(figsize=(8, 7))
ax = f.add_subplot(221)
x,y = beamx,beamy
plt.scatter(beamx, beamy, facecolors='none', edgecolors='k')
plt.xlabel(r'$Beam\ X\ (\mu m)$')
plt.ylabel(r'$Beam\ Y\ (\mu m)$')
plt.xlim(x.min() - 0.1*(x.max() - x.min()) ,x.max() + 0.1*(x.max() - x.min()))
plt.ylim(y.min() - 0.1*(y.max() - y.min()) ,y.max() + 0.1*(y.max() - y.min()))
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')

ax = f.add_subplot(222)
x,y = ipmy,beamy
plt.scatter(ipmy, beamy, facecolors='none', edgecolors='k')
plt.ylabel(r'$Beam\ Y\ (\mu m)$')
plt.xlabel(r'$IPM_Y\ \left( \frac{IPM_T - IPM_B}{IPM_T + IPM_B} \right)$', fontsize=16, labelpad=14.)
plt.xlim(x.min() - 0.1*(x.max() - x.min()) ,x.max() + 0.1*(x.max() - x.min()))
plt.ylim(y.min() - 0.1*(y.max() - y.min()) ,y.max() + 0.1*(y.max() - y.min()))
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')

ax = f.add_subplot(223)
x,y = beamx,ipmx
plt.scatter(beamx, ipmx, facecolors='none', edgecolors='k')
plt.xlabel(r'$Beam\ X\ (\mu m)$')
plt.ylabel(r'$IPM_X\ \left( \frac{IPM_R - IPM_L}{IPM_R + IPM_L}\right)$', fontsize=16)
plt.xlim(x.min() - 0.1*(x.max() - x.min()) ,x.max() + 0.1*(x.max() - x.min()))
plt.ylim(y.min() - 0.1*(y.max() - y.min()) ,y.max() + 0.1*(y.max() - y.min()))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')

ax = f.add_subplot(224)
x,y = ipmy,ipmx
plt.scatter(ipmy, ipmx, facecolors='none', edgecolors='k')
plt.ylabel(r'$IPM_Y\ \left( \frac{IPM_T - IPM_B}{IPM_T + IPM_B} \right)$', fontsize=16)
plt.xlabel(r'$IPM_X\ \left( \frac{IPM_R - IPM_L}{IPM_R + IPM_L} \right)$', fontsize=16)
plt.xlim(x.min() - 0.1*(x.max() - x.min()) ,x.max() + 0.1*(x.max() - x.min()))
plt.ylim(y.min() - 0.1*(y.max() - y.min()) ,y.max() + 0.1*(y.max() - y.min()))
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()

plt.tight_layout()
#plt.show()

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../renders/ipm_position.png')
plt.savefig(filename, fmt='png', dpi=600)
