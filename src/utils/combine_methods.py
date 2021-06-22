#!/usr/bin/env python3

from . import pdbtools, pdb_resdepth, matrice_distances
from . import Lennard_Jones, electrostatic
from . import shape_complement, knowledge


import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

def combine_score(pdbfile, recepChain, ligChain, statpotrun = True, vdwrun = True, electrorun = True, shaperun = True, pH = True, depth = "msms", dist = 8.6):
    combined_dict = {}
    my_struct = pdbtools.read_pdb(pdbfile)
    depth_dict = pdb_resdepth.calculate_resdepth(structure=my_struct, pdb_filename=pdbfile, method= depth)
    distmat = matrice_distances.calc_distance_matrix(structure=my_struct, depth= depth_dict, chain_R=recepChain, chain_L=ligChain, dist_max=dist, method=depth)

    combined_dict["pdb"] = pdbfile.split("/")[-1]
    if statpotrun:
        statpot = knowledge.parse_distance_mat(distmat, method=["glaser"])
        combined_dict["statpot"] = statpot
    if vdwrun:
        vdw = Lennard_Jones.lennard_jones(dist_matrix=distmat)
        combined_dict["vdw"] = vdw
    if electrorun:
        electro = electrostatic.electrostatic(inter_resid_dict=distmat, pH=pH)
        combined_dict["electro"] = electro
    if shaperun:
        shape = shape_complement.runshape(structure=my_struct, recepChain=recepChain, depth_dict=depth_dict, ligChain=ligChain, method=depth)
        combined_dict["shape"] = shape

    return(combined_dict)


"""
import sys

if __name__ == '__main__':
    myfile = sys.argv[1]
    recepChain = sys.argv[2].split(",")
    ligChain = sys.argv[3].split(",")
    mydict = combine_score(pdbfile=myfile, recepChain=recepChain, ligChain=ligChain)
    print(mydict)
"""