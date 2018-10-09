import pandas as pd
import numpy as np
import crystal
from subprocess import call


onFN  = "../1ubq-flip.pdb.hkl"
offFN = "../1ubq.pdb.hkl"
pdbFN = "../1ubq.pdb"


Fon  = crystal.crysstal().read_hkl(onFN)
Foff = crystal.crysstal().read_hkl(offFN)
header = ''.join(open(offFN).readlines()[:3] ) + "DECLare NAME=F  DOMAin=RECIprocal TYPE=COMPLEX END\n"


call("phenix.reflection_file_converter {} --mtz Foff.mtz".format(offFN).split())
call("phenix.maps Foff.mtz {}".format(pdbFN).split())
call('phenix.reflection_file_converter _map_coeffs.mtz --label=2FOFCWT,PH2FOFCWT --cns=phases'.split())

F = pd.read_csv("phases.cns", delim_whitespace=True, skiprows=6, usecols=[1,2,3,5,6], names=['H', 'K', 'L', 'Foff', 'Phase'], index_col=['H', 'K', 'L'])

F['Fon'] = Fon['F']
F['DeltaF'] = ((Fon - Foff)/Foff)['F']


for i in range(101):
    cutoff = np.percentile(np.abs(F['DeltaF']), i)
    F[np.abs(F['DeltaF']) < cutoff] = 0.

    outFN = "{:03d}.hkl".format(i)
    with open(outFN, 'w') as out:
        out.write(header)
        for hkl,r in F.iterrows():
            out.write("INDE {} {} {} F= {:0.3f} {}\n".format(hkl[0], hkl[1], hkl[2], r['DeltaF'], r['Phase']))

    call("phenix.reflection_file_converter {} --mtz {:03d}.mtz".format(outFN, i).split())



"""
call("phenix.fobs_minus_fobs_map f_obs_1_file=Fon_{:03d}.hkl f_obs_2_file=Foff_{:03d}.hkl {} --silent".format(i, i, pdbFN).split())
call("mv FoFoPHFc.mtz {:03d}.mtz".format(i).split())
"""
