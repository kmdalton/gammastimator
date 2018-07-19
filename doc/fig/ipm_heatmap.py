import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import ipm,scatter


mpl.rcParams['font.size'] = 12.
mpl.rcParams['mathtext.default'] = 'sf'

vmin,vmax = 0., 5.
#colormapname = 'Greys_r'
colormapname = 'viridis'
points = 500
contours = 1000
beamx,beamy = 0., 0.
ko = 12398. #X-ray energy in eV

X,Y = np.linspace(-17000, 17000, points), np.linspace(-17000, 17000, points)
X,Y = np.meshgrid(X, Y)
theta,phi = scatter.transform_spherical(X-beamx, Y-beamy, ipm.film_distance)

Z = 1e28*(
    4.*scatter.differential_intensity(theta, phi, ko, 7) + \
    3.*scatter.differential_intensity(theta, phi, ko, 14)
    )

print(Z[int(points/2), int(points/2)])

mask = (X >= -15000.) & (X <= -5000.) & (Y >= -5000.) & (Y <= 5000.)  | \
       (X >= -5000.)  & (X <= 5000.)  & (Y >= -15000.)& (Y <= -5000.) | \
       (X >= -5000.)  & (X <= 5000.)  & (Y >= 5000.)  & (Y <= 15000.) | \
       (X >= 5000.)   & (X <= 15000.) & (Y >= -5000.) & (Y <= 5000.)  

cmap = plt.get_cmap(colormapname)
norm = plt.Normalize(vmin, vmax)
sm   = mpl.cm.ScalarMappable(norm, cmap)
sm.set_array(Z)

plt.figure(figsize=(6,5))
plt.contourf(X, Y, Z*mask, contours, cmap=cmap, norm=norm)

"""
#Plot panel boundaries
plt.vlines(-5000, -15000, 15000, 'w')
plt.vlines(5000, -15000, 15000,  'w')

plt.vlines(-15000, -5000, 5000,  'w')
plt.vlines( 15000, -5000, 5000,  'w')

plt.hlines(-5000, -15000, 15000, 'w')
plt.hlines(5000, -15000, 15000,  'w')

plt.hlines(-15000, -5000, 5000, 'w')
plt.hlines( 15000, -5000, 5000, 'w')
"""

plt.colorbar(sm, label=r"$\frac{d \sigma}{d \Omega}\ (barns\ sr^{-1}\ molecule^{-1})$")
plt.title(r"$Differential\ Cross\ Section\ of\ Si_3 N_4$")

#Plot Beam Center
plt.scatter(beamx, beamy, s=50., facecolors='none', edgecolors='w')

plt.axis((-17000, 17000, -17000, 17000))
plt.xlabel(r'$Xpos\ (mm)$')
plt.ylabel(r'$Ypos\ (mm)$')

plt.xticks([-15000, -10000, -5000, 0, 5000, 10000, 15000], [-15, -10, -5, 0, 5, 10, 15])
plt.yticks([-15000, -10000, -5000, 0, 5000, 10000, 15000], [-15, -10, -5, 0, 5, 10, 15])

#plt.show()

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../renders/ipm_heatmap.png')
plt.savefig(filename, fmt='png', dpi=600)

