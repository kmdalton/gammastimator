from matplotlib import pyplot as plt
import matplotlib as mpl
import simulate


inFN = "1ubq.pdb.hkl"
x = simulate.crystal(inFN)
#x.orientrandom()


f = plt.figure(figsize=(8,8))
for i in range(361):
    F = x.reflections()
    sizes = 50.*F['F']**2 / ((F['F']**2).max())
    plt.scatter(F['X'], F['Y'], s=sizes, c=sizes)
    plt.xlim(-60., 60.)
    plt.ylim(-60., 60.)
    plt.title(r"$\phi = {}\deg$".format(i))
    plt.xlabel(r"X-position (mm)")
    plt.ylabel(r"Y-position (mm)")
    plt.savefig('{:03}.png'.format(i))
    plt.clf()
    x.rotate(1)
