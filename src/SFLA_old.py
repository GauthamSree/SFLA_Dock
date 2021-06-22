import os
import glob
import concurrent.futures

# import quaternion as quat
# from numpy.random import Generator, PCG64
import Bio.PDB
import shutil
from pyquaternion import Quaternion

# from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import scipy.spatial
import numpy as np
import argparse
import random
import re
import sys
import math
from Bio.PDB import *
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import min_dist
from meetdock import *

from lib import pdbtools
from lib import pdb_resdepth
from lib import matrice_distances
from lib import Lennard_Jones
from lib import electrostatic

# from surface import *
p=PDBParser()

recognized_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                           'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'NH', 'OC']
atom_types = [['N'], ['CA'], ['C'], ['O'], ['GLYCA'],
                  ['ALACB', 'ARGCB', 'ASNCB', 'ASPCB', 'CYSCB', 'GLNCB', 'GLUCB', 'HISCB', 'ILECB', 'LEUCB', 'LYSCB',
                   'METCB', 'PHECB', 'PROCB', 'PROCG', 'PROCD', 'THRCB', 'TRPCB', 'TYRCB', 'VALCB'],
                  ['LYSCE', 'LYSNZ'], ['LYSCD'], ['ASPCG', 'ASPOD1', 'ASPOD2', 'GLUCD', 'GLUOE1', 'GLUOE2'],
                  ['ARGCZ', 'ARGNH1', 'ARGNH2'],
                  ['ASNCG', 'ASNOD1', 'ASNND2', 'GLNCD', 'GLNOE1', 'GLNNE2'], ['ARGCD', 'ARGNE'],
                  ['SERCB', 'SEROG', 'THROG1', 'TYROH'],
                  ['HISCG', 'HISND1', 'HISCD2', 'HISCE1', 'HISNE2', 'TRPNE1'], ['TYRCE1', 'TYRCE2', 'TYRCZ'],
                  ['ARGCG', 'GLNCG', 'GLUCG', 'ILECG1', 'LEUCG', 'LYSCG', 'METCG', 'METSD', 'PHECG', 'PHECD1', 'PHECD2',
                   'PHECE1', 'PHECE2', 'PHECZ', 'THRCG2', 'TRPCG', 'TRPCD1', 'TRPCD2', 'TRPCE2', 'TRPCE3', 'TRPCZ2',
                   'TRPCZ3', 'TRPCH2', 'TYRCG', 'TYRCD1', 'TYRCD2'],
                  ['ILECG2', 'ILECD1', 'ILECD', 'LEUCD1', 'LEUCD2', 'METCE', 'VALCG1', 'VALCG2'], ['CYSSG']]

rng = np.random.default_rng(0)

# return residue-wise representation
def chaindef(file, rec_chain):

    structure = p.get_structure("1bth", file)
    coordinatesr = np.empty((0, 3))
    tobi_residuesr = []
    residue_id = []
    boundary_residue_coord = np.empty((0, 3))
    atom_coord = np.empty((0, 3))
    boundary_residue_id = []
    boundary_residue_name = []
    # rcc=0
    for model in structure:
        surface = get_surface(model)
        for chain in model:
            if chain.id in rec_chain:
                for residue in chain:
                    # print('hi')
                    cx = 0.0
                    cy = 0.0
                    cz = 0.0
                    count = 0
                    residue_index = recognized_residues.index(residue.get_resname())
                    atom_set = np.empty((0, 3))
                    for atom in residue:
                        if not atom.name == "H":
                            ax = atom.get_coord()[0]
                            ay = atom.get_coord()[1]
                            az = atom.get_coord()[2]
                            atom_set = np.append(atom_set, [atom.get_coord()], axis=0)
                            atom_coord = np.append(atom_coord, [atom.get_coord()], axis=0)
                            cur_atom = residue.get_resname() + atom.name
                            for typ in atom_types:
                                if cur_atom in typ or atom.name in ["N", "CA", "C", "O"]:  # typ:#atom.name now added
                                    cx += ax
                                    cy += ay
                                    cz += az
                                    count += 1
                                else:
                                    pass
                    cx /= float(count)
                    cy /= float(count)
                    cz /= float(count)
                    coordinatesr = np.append(coordinatesr, [[cx, cy, cz]], axis=0)
                    # rcc+=1
                    tobi_residuesr.append(residue_index)
                    residue_id.append(str(residue.get_id()[1]) + residue.get_id()[2])
                    fji = 0  # check whether any of of the atoms in the resdue are at a distance 3 A from surface
                    for ji in range(len(atom_set)):
                        if min_dist(atom_set[ji], surface) < 2:
                            fji = 1
                            break
                    if fji == 1:
                        boundary_residue_coord = np.append(boundary_residue_coord, [[cx, cy, cz]], axis=0)
                        # boundary_atom_name.append(atom.name)
                        boundary_residue_id.append(str(residue.get_id()[1]) + residue.get_id()[2])
                        boundary_residue_name.append(residue.get_resname())
    # print(rcc)
    return boundary_residue_coord, boundary_residue_name, boundary_residue_id, atom_coord


# compute shape descriptor
def findPointNormals(points, numNeighbours, viewPoint, residue_id, residue_name, f):
    viewPoint = [float(x) for x in viewPoint]
    nbrs = NearestNeighbors(n_neighbors=numNeighbours + 1, algorithm="kd_tree").fit(points)
    distances, indices = nbrs.kneighbors(points)
    n = []  # indices[:,2:]
    [n.append(indices[i][1:].tolist()) for i in range(0, len(indices))]

    #%find difference in position from neighbouring points
    n = np.asarray(n).flatten("F")
    p = np.tile(points, (numNeighbours, 1)) - points[n]
    x = np.zeros((3, len(points), numNeighbours))
    for i in range(0, 3):
        for j in range(0, len(points)):
            for k in range(0, numNeighbours):
                x[i, j, k] = p[k * len(points) + j, i]
    p = x
    C = np.zeros((len(points), 6))
    C[:, 0] = np.sum(np.multiply(p[0], p[0]), axis=1)
    C[:, 1] = np.sum(np.multiply(p[0], p[1]), axis=1)
    C[:, 2] = np.sum(np.multiply(p[0], p[2]), axis=1)
    C[:, 3] = np.sum(np.multiply(p[1], p[1]), axis=1)
    C[:, 4] = np.sum(np.multiply(p[1], p[2]), axis=1)
    C[:, 5] = np.sum(np.multiply(p[2], p[2]), axis=1)
    C = np.divide(C, numNeighbours)
    normals = np.zeros((len(points), 3))
    curvature = np.zeros((len(points), 1))
    for i in range(0, len(points)):
        Cmat = [
            [C[i, 0], C[i, 1], C[i, 2]],
            [C[i, 1], C[i, 3], C[i, 4]],
            [C[i, 2], C[i, 4], C[i, 5]],
        ]
        [value, vector] = np.linalg.eigh(Cmat)
        [lam, k] = min(value), value.tolist().index(min(value))
        normals[i, :] = vector[:, k]  # np.transpose(vector[:,k])
        curvature[i] = lam / sum(value)

    return normals, curvature


'''
def rotate(origin, point, angle, seed):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.        , [0,0,0,1]
    """
    sx, sy, sz = seed
    ox, oy, oz = origin
    px, py, pz = point

    return origin + np.dot(np.array([[math.cos(angle)+pow(sx,2)*(1-math.cos(angle)), sx*sy*(1-math.cos(angle))-sz*math.sin(angle), sx*sz*(1-math.cos(angle))+sy*math.sin(angle)], \
      [sy*sx*(1-math.cos(angle))+sz*math.sin(angle), math.cos(angle)+pow(sy,2)*(1-math.cos(angle)), sy*sz*(1-math.cos(angle))-sx*math.sin(angle)], \
                       [sz*sx*(1-math.cos(angle))-sy*math.sin(angle), sz*sy*(1-math.cos(angle))+sx*math.sin(angle), math.cos(angle)+pow(sz,2)*(1-math.cos(angle))]]), np.subtract(point, origin))
'''

depth = "msms"
dist = 8.6
pH = 7
# pose generation and score calculation
def do_something(args):

    output_file = "out" + str(args[1]) + ".pdb"
    out = open(os.path.join(mypath, output_file), "w")
    in1 = open(inp2, "r")
    in2 = open(inp1, "r")
    for line in in1:
        if "ATOM" in line:
            out.write(line)
    indexing = 0
    new_co = args[0]
    for line in in2:
        if "ATOM" in line:
            # print(line)
            l = line.split()
            l[0] = l[0].ljust(5)
            l[1] = l[1].rjust(5)
            l[2] = l[2].ljust(3)
            l[3] = l[3].ljust(3)
            l[4] = line[21]
            l[5] = ("%4d" % (int(line[22:26]))).rjust(4)
            l[6] = ("%8.3f" % (float(new_co[indexing][0]))).rjust(8)
            l[7] = ("%8.3f" % (float(new_co[indexing][1]))).rjust(8)
            l[8] = ("%8.3f" % (float(new_co[indexing][2]))).rjust(8)
            out.write(
                "{0} {1}  {2} {3} {4}{5}    {6}{7}{8}".format(
                    l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]
                )
            )
            out.write("\n")
            indexing += 1
    out.close()
    # print("depth ok")

    pdbfile = os.path.join(mypath, output_file)
    my_struct = pdbtools.read_pdb(pdbfile)
    try:

        depth_dict = pdb_resdepth.calculate_resdepth(
            structure=my_struct, pdb_filename=pdbfile, method=depth
        )
    except:
        os.remove(os.path.join(mypath, output_file))
        return
    distmat = matrice_distances.calc_distance_matrix(
        structure=my_struct,
        depth=depth_dict,
        chain_R=rec_chain,
        chain_L=lig_chain,
        dist_max=dist,
        method=depth,
    )

    vdw = Lennard_Jones.lennard_jones(dist_matrix=distmat)
    electro = electrostatic.electrostatic(inter_resid_dict=distmat, pH=pH)
    score = vdw + electro

    return score, args[1], args[2], args[3]

def find_score(args):
    output_file='out'+str(args[1])+'.pdb'
    shape, electro, jones, proba = True, True, True, False
    pH = 7
    dist = 8.6
    with open(os.path.join(mypath, output_file),'w') as out:
        in1 = open(inp2, "r")
        in2 = open(inp1, "r")
        for line in in1:
            if "ATOM" in line:
                out.write(line)
        indexing = 0
        new_co = args[0]
        for line in in2:
            if "ATOM" in line:
                # print(line)
                l = line.split()
                l[0] = l[0].ljust(5)
                l[1] = l[1].rjust(5)
                l[2] = l[2].ljust(3)
                l[3] = l[3].ljust(3)
                l[4] = line[21]
                l[5] = ("%4d" % (int(line[22:26]))).rjust(4)
                l[6] = ("%8.3f" % (float(new_co[indexing][0]))).rjust(8)
                l[7] = ("%8.3f" % (float(new_co[indexing][1]))).rjust(8)
                l[8] = ("%8.3f" % (float(new_co[indexing][2]))).rjust(8)
                out.write(
                    "{0} {1}  {2} {3} {4}{5}    {6}{7}{8}".format(
                        l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]
                    )
                )
                out.write("\n")
                indexing += 1

    pdbfile = os.path.join(mypath, output_file)
    res = cm.combine_score(pdbfile, recepChain=rec_chain, ligChain=lig_chain, statpotrun=proba, vdwrun=jones, electrorun=electro, shaperun=shape, pH=pH, depth=depth, dist=dist)
    mydf = pd.DataFrame(res,  index=[0])
    mydf = mydf.set_index('pdb')
    score = tm.tm_score(mydf, execdir='.')
    return float(score['tm_score_prediction']), args[1], args[2], args[3]

# preprocessing
def pdbpre(file1):

    pdb_in = open(os.path.join(args.pdb, file1), "r")
    # print(file1)
    out = open(file1 + "1.pdb", "w")
    atmno = 1
    resno = 0
    res = ""
    fr = ""
    l = [""] * 11
    for line in pdb_in:
        if "ATOM" in line[0:4]:
            li = line.split()
            l[0] = li[0].ljust(6)
            l[1] = str(atmno).rjust(4)
            l[2] = li[2].ljust(3)
            l[3] = li[3].ljust(3)
            l[4] = line[21]
            if fr != line[21]:
                atmno = 1
                resno = 0
                res = ""
                fr = line[21]
            if line[22:26] == res:
                l[5] = ("%4d" % (int(resno))).rjust(4)
            else:
                resno += 1
                res = line[22:26]
                l[5] = ("%4d" % (int(resno))).rjust(4)
            # if len(l[6])>10:
            l[6] = ("%8.3f" % (float(line[29:38]))).rjust(8)
            l[7] = ("%8.3f" % (float(line[38:46]))).rjust(8)
            l[8] = ("%8.3f" % (float(line[46:54]))).rjust(8)
            l[9] = ("%6.2f" % (float(line[55:60]))).rjust(6)
            l[10] = ("%6.2f" % (float(line[60:66]))).ljust(6)
            out.write(
                "{0} {1}  {2} {3} {4}{5}    {6}{7}{8}{9}{10}".format(
                    l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10]
                )
            )
            out.write("\n")
            atmno += 1
        out.close()

    return file1 + "1.pdb"


def sort_frog(mplx_no):
    sorted_fitness = np.array(sorted(StructInfo, key = lambda x: StructInfo[x][0]))

    memeplexes = np.zeros((mplx_no, int(frogs/mplx_no)))

    for j in range(memeplexes.shape[1]):
        for i in range(mplx_no):
            memeplexes[i, j] = sorted_fitness[i + (mplx_no*j)] 
    return memeplexes

def shuffle_memeplexes(memeplexes):
    """Shuffles the memeplexes and sorting them.
    
    Arguments:
        frogs {numpy.ndarray} -- All the frogs
        memeplexes {numpy.ndarray} -- The memeplexes
    
    Returns:
        None
    """
    mplx_no = memeplexes.shape[0]
    temp = memeplexes.flatten()
    temp = np.array(sorted(temp, key = lambda x: StructInfo[x][0]))
    for j in range(memeplexes.shape[1]):
        for i in range(mplx_no):
            memeplexes[i, j] = temp[i + (mplx_no*j)]

# def levy(d):
# global seedc
"""
lamda = 1.5
sigma = (
    math.gamma(1 + lamda)
    * math.sin(math.pi * lamda / 2)
    / (math.gamma((1 + lamda) / 2) * lamda * (2 ** ((lamda - 1) / 2)))
) ** (1 / lamda)
"""

sys.path.insert(0, "./support")
parser = argparse.ArgumentParser(description="Molecular conformer generator")
parser.add_argument("-pdb", required=True, help="sdf output file")
# parser.add_argument('-lpdb', required=True, help='sdf input file')
# parser.add_argument('-recchain', required=True, help='sdf output file')
# parser.add_argument('-ligchain', required=True, help='sdf output file')

parser.add_argument("-n", type=int, required=False, help="number of conformers")
# parser.add_argument('-rtpre', type=float, required=False, help='rms threshold pre optimization')
# parser.add_argument('-rtpost', type=float, required=False, help='rms threshold post minimization')

args = parser.parse_args()
# print args.pdb
pdb0 = args.pdb.split("/")[-1].split("_")[0]
pdb1 = args.pdb.split("/")[-1].split("_")[1].split(":")
rpdb = pdb1[0] + "_model_st.pdb"
lpdb = pdb1[1] + "_model_st.pdb"
lig_chain = []
rec_chain = []
for i in pdb1[0]:
    rec_chain.append(i)
for i in pdb1[1]:
    lig_chain.append(i)

inp1 = pdbpre(lpdb)
inp2 = pdbpre(rpdb)
n = args.n
# n=200


frogs = 50  ## No of frogs (population)
StructInfo = {}
init = 0
mypath = "poses/"
N_iter = 10  ## No of iternations

def generate_one_frog(init):
    Quater = [0, 0, 0, 0]
    recRandIdx = rng.integers(0, rec_coord.shape[0] - 1)
    ligRandIdx = rng.integers(0, lig_coord.shape[0] - 1)
    axis = rec_coord[recRandIdx]
    a = rec_normal[recRandIdx]
    b = lig_normal[ligRandIdx]
    
    dotProduct = np.dot(a, b)
    theta = np.arccos(dotProduct) * 2 - np.pi
    
    Quater = Quaternion(axis=a, angle=theta)
    
    final = np.array([Quater.rotate(i) for i in lig_atom])
    args = [[final, init, Quater, -1]]
    return args

## TODO: Local search

def local_search_one_memeplex(im):
    for iN in range(N):
        rValue = rng.random(FrogsEach) * weights # random value with probability weights
        subindex = np.sort(np.argsort(rValue)[::-1][0:q]) # index of selected frogs in memeplex
        submemeplex = memeplexes[im][subindex] 
        
        #--- Improve the worst frog's position ---#
        # Learn from local best Pb #
        Pb = StructInfo[int(submemeplex[0])] # mark the best frog in submemeplex
        Pw = StructInfo[int(submemeplex[q-1])] # mark the worst frog in memeplex
        
        S = rng.random() * (Pb[1] - Pw[1]) 
        Uq = Pw[1] + S
        
        globStep = False
        censorship = False
        
        # Check feasible space and the performance #
        if Omega[0] <= min(Uq) and max(Uq) <= Omega[1]: # check feasible space
            final = np.array([Uq.rotate(i) for i in lig_atom])  
            results = do_something([final, init+1, Uq, im])
            
            if results[0] > Pw[0]:
                globStep = True
        
        if globStep:
            S = rng.random() * (Frog_gb[1] - Pw[1])
            for i in range(4):
                if S[i] > 0:
                    S[i] = min(S[i],max_step)
                else:
                    S[i] = max(S[i],-max_step)
            Uq = Pw[1] + S
            
            if Omega[0] <= min(Uq) and max(Uq) <= Omega[1]: # check feasible space
                final = np.array([Uq.rotate(i) for i in lig_atom])  
                results = do_something([final, init+1, Uq, im])
                if results[0] > Pw[0]:
                    censorship = True
            else:
                censorship = True
        
        if censorship:
            recRandIdx = rng.integers(0, rec_coord.shape[0] - 1)
            ligRandIdx = rng.integers(0, lig_coord.shape[0] - 1)

            axis = rec_coord[recRandIdx]
            a = rec_normal[recRandIdx]
            b = lig_normal[ligRandIdx]
            
            dotProduct = np.dot(a, b)
            theta = np.arccos(dotProduct) * 2 - np.pi
            Quater = Quaternion(axis=a, angle=theta)
            final = np.array([Quater.rotate(i) for i in lig_atom])
            results = do_something([final, init+1, Quater, im])            
        
        
        #StructInfo[im] = [results[0], results[2]]
        shutil.move(os.path.join('poses/', 'out'+str(init+1)+'.pdb'), os.path.join('poses/', 'out'+ str(submemeplex[q-1]) + '.pdb'))
        StructInfo[int(submemeplex[q-1])] = [results[0], results[2]]
        memeplexes[im] = memeplexes[im][np.argsort(memeplexes[im])]

if True:

    lig_coord, lig_res, lig_res_id, lig_atom = chaindef(inp1, lig_chain)
    rec_coord, rec_res, rec_res_id, rec_atom = chaindef(inp2, rec_chain)

    rec_normal, rec_curve = findPointNormals(
        rec_coord, 20, [0, 0, 0], rec_res_id, rec_res, "r"
    )
    lig_normal, lig_curve = findPointNormals(
        lig_coord, 20, [0, 0, 0], lig_res_id, lig_res, "r"
    )
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        Doargs = []
        for i in range(frogs):
            Quater = [0, 0, 0, 0]
            recRandIdx = rng.integers(0, rec_coord.shape[0] - 1)
            ligRandIdx = rng.integers(0, lig_coord.shape[0] - 1)
            axis = rec_coord[recRandIdx]
            a = rec_normal[recRandIdx]
            b = lig_normal[ligRandIdx]
            
            dotProduct = np.dot(a, b)
            theta = np.arccos(dotProduct) * 2 - np.pi
            
            Quater = Quaternion(axis=a, angle=theta)
            
            final = np.array([Quater.rotate(i) for i in lig_atom])
            
            Doargs += [[final, init, Quater, -1]]
            init += 1
        results = executor.map(do_something, Doargs)
        for r in results:
            if r:
                StructInfo[r[1]] = [r[0], r[2]]

    # TODO: SORT AND ADD TO MEMEPLEXES
    memeplexes = sort_frog(10)    
    
    for _ in range(N_iter):
        # TODO: LOCAL SEARCH
        Frog_gb = StructInfo[int(memeplexes[0][0])]
        FrogsEach = int(frogs/len(memeplexes)) #the number of frogs in each memeplex
        weights = [2*(FrogsEach+1-j)/(FrogsEach*(FrogsEach+1)) for j in range(1, FrogsEach+1)] 

        Omega = [np.amin(rec_normal), np.amax(rec_normal)]
        max_step = (Omega[1]-Omega[0])/2 # maximum step size
        q = 3 # int, the number of frogs in submemeplex -- CHANGE
        N = 1
        for im in range(len(memeplexes)):
            for iN in range(N):
                rValue = rng.random(FrogsEach) * weights # random value with probability weights
                subindex = np.sort(np.argsort(rValue)[::-1][0:q]) # index of selected frogs in memeplex
                submemeplex = memeplexes[im][subindex] 
                
                #--- Improve the worst frog's position ---#
                # Learn from local best Pb #
                Pb = StructInfo[int(submemeplex[0])] # mark the best frog in submemeplex
                Pw = StructInfo[int(submemeplex[q-1])] # mark the worst frog in memeplex
                
                S = rng.random() * (Pb[1] - Pw[1]) 
                Uq = Pw[1] + S
                
                globStep = False
                censorship = False
                
                # Check feasible space and the performance #
                if Omega[0] <= min(Uq) and max(Uq) <= Omega[1]: # check feasible space
                    final = np.array([Uq.rotate(i) for i in lig_atom])  
                    results = do_something([final, init+1, Uq, im])
                    
                    if results[0] > Pw[0]:
                        globStep = True
                
                if globStep:
                    S = rng.random() * (Frog_gb[1] - Pw[1])
                    for i in range(4):
                        if S[i] > 0:
                            S[i] = min(S[i],max_step)
                        else:
                            S[i] = max(S[i],-max_step)
                    Uq = Pw[1] + S
                    
                    if Omega[0] <= min(Uq) and max(Uq) <= Omega[1]: # check feasible space
                        final = np.array([Uq.rotate(i) for i in lig_atom])  
                        results = do_something([final, init+1, Uq, im])
                        if results[0] > Pw[0]:
                            censorship = True
                    else:
                        censorship = True
                
                if censorship:
                    recRandIdx = rng.integers(0, rec_coord.shape[0] - 1)
                    ligRandIdx = rng.integers(0, lig_coord.shape[0] - 1)

                    axis = rec_coord[recRandIdx]
                    a = rec_normal[recRandIdx]
                    b = lig_normal[ligRandIdx]
                    
                    dotProduct = np.dot(a, b)
                    theta = np.arccos(dotProduct) * 2 - np.pi
                    Quater = Quaternion(axis=a, angle=theta)
                    final = np.array([Quater.rotate(i) for i in lig_atom])
                    results = do_something([final, init+1, Quater, im])            
                
                
                #StructInfo[im] = [results[0], results[2]]
                shutil.move(os.path.join('poses/', 'out'+str(init+1)+'.pdb'), os.path.join('poses/', 'out'+ str(submemeplex[q-1]) + '.pdb'))
                StructInfo[int(submemeplex[q-1])] = [results[0], results[2]]
                memeplexes[im] = memeplexes[im][np.argsort(memeplexes[im])]
        
        # SHUFFLE
        memeplexes = shuffle_memeplexes(memeplexes)
        Frog_gb = StructInfo[int(memeplexes[0][0])]
        
        # TODO: CONVERGENCE CHECK
    
    with open("bestenergy.txt", "w") as be:
        for m in memeplexes:
            besti = m[0]
            minVal = StructInfo[int(m[0])][0]
            be.write(str(besti) + "\t" + str(minVal) + "\n")

    for fname in glob.glob("/tmp/tmp*"):
        try:
            os.remove(fname)
        except:
            pass

    directory = "native_" + pdb0
    path = os.path.join("./", directory)
    os.mkdir(path)
    shutil.move(
                os.path.join("poses/", "out" + str(memeplexes[0][0]) + ".pdb"), directory
            )  # move best global frog to native folder
