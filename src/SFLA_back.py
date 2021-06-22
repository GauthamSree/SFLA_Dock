import os, glob, sys, math
import concurrent.futures
import argparse

import shutil

# from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from Bio.PDB import *
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import min_dist
from pyquaternion import Quaternion

from utils import pdbtools
from utils import pdb_resdepth
from utils import matrice_distances
from utils import Lennard_Jones
from utils import electrostatic
from utils import combine_methods as cm
from utils import tm_score as tm

# from surface import *
p = PDBParser()

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


class SFLA:
    def __init__(self, frogs, mplx_no, n_iter, N, q):
        self.frogs = frogs
        self.mplx_no = mplx_no
        self.FrogsEach = int(self.frogs/self.mplx_no)
        self.weights = [2*(self.FrogsEach+1-j)/(self.FrogsEach*(self.FrogsEach+1)) for j in range(1, self.FrogsEach+1)] 
        self.structinfo = {}
        self.init = 0
        self.mypath ='poses/'
        self.n_iter = n_iter
        self.N = N
        self.q = q
    
    def chaindef(self, file, rec_chain):
        structure = p.get_structure('1bth', file)
        coordinatesr = np.empty((0,3))
        tobi_residuesr = []
        residue_id = []
        boundary_residue_coord = np.empty((0,3))
        atom_coord=np.empty((0,3))
        boundary_residue_id=[]
        boundary_residue_name=[]
        for model in structure:
            surface = get_surface(model)
            for chain in model:
                if chain.id in rec_chain:
                    for residue in chain:
                        cx = 0.0
                        cy = 0.0
                        cz = 0.0
                        count = 0
                        residue_index=recognized_residues.index(residue.get_resname())
                        atom_set=np.empty((0,3))
                        for atom in residue:
                            if  not atom.name=='H':
                                ax=atom.get_coord()[0]
                                ay=atom.get_coord()[1]
                                az=atom.get_coord()[2]
                                atom_set=np.append(atom_set,[atom.get_coord()], axis=0)
                                atom_coord=np.append(atom_coord,[atom.get_coord()], axis=0)
                                cur_atom=residue.get_resname()+atom.name
                                for typ in atom_types:
                                    if  cur_atom in typ or atom.name in ['N','CA','C','O']:	#typ:#atom.name now added
                                        cx += ax
                                        cy += ay
                                        cz += az
                                        count += 1
                                    else:
                                        pass
                        cx/= float(count)
                        cy/= float(count)
                        cz/= float(count)
                        coordinatesr=np.append(coordinatesr,[[cx, cy, cz]], axis=0)
                        #rcc+=1
                        tobi_residuesr.append(residue_index)
                        residue_id.append(str(residue.get_id()[1])+residue.get_id()[2])
                        fji=0     #check whether any of of the atoms in the resdue are at a distance 3 A from surface
                        for ji in range(len(atom_set)):
                            if min_dist(atom_set[ji], surface) < 2:
                                fji=1
                                break
                        if fji==1:
                            boundary_residue_coord=np.append(boundary_residue_coord,[[cx, cy, cz]],axis=0)
                            #boundary_atom_name.append(atom.name)
                            boundary_residue_id.append(str(residue.get_id()[1])+residue.get_id()[2])
                            boundary_residue_name.append(residue.get_resname())
        
        return boundary_residue_coord,boundary_residue_name, boundary_residue_id, atom_coord
    
    def findPointNormals(self, points, numNeighbours, viewPoint, residue_id, residue_name, f):
        nbrs = NearestNeighbors(n_neighbors=numNeighbours+1, algorithm='kd_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        n = []
        [n.append(indices[i][1:].tolist()) for i in range(0,len(indices))]

        # find difference in position from neighbouring points
        n=np.asarray(n).flatten('F')    
        p = np.tile(points,(numNeighbours,1)) - points[n]
        x=np.zeros((3,len(points),numNeighbours))
        for i in range(0,3):
            for j in range(0,len(points)):
                for k in range(0,numNeighbours):
                    x[i,j,k]=p[k*len(points)+j,i]
        p = x
        C = np.zeros((len(points),6))
        C[:,0]= np.sum(np.multiply(p[0],p[0]),axis=1)
        C[:,1]= np.sum(np.multiply(p[0],p[1]),axis=1)
        C[:,2]= np.sum(np.multiply(p[0],p[2]),axis=1)
        C[:,3]= np.sum(np.multiply(p[1],p[1]),axis=1)
        C[:,4]= np.sum(np.multiply(p[1],p[2]),axis=1)
        C[:,5]= np.sum(np.multiply(p[2],p[2]),axis=1)
        C = np.divide(C, numNeighbours)
        normals = np.zeros((len(points),3))
        curvature = np.zeros((len(points),1))
        for i in range(0,len(points)):
            Cmat = [[C[i,0], C[i,1] ,C[i,2]], [C[i,1], C[i,3], C[i,4]], [C[i,2], C[i,4], C[i,5]]]
            [value,vector] = np.linalg.eigh(Cmat)
            [lam,k] = min(value), value.tolist().index(min(value))
            normals[i,:] = vector[:,k] #np.transpose(vector[:,k])
            curvature[i]= lam / sum(value)

        return normals, curvature

    def find_score(self, args):
        output_file='out' + str(args[1]) + '.pdb'
        pH = 7
        dist = 8.6
        depth = "msms"
        with open(os.path.join(self.mypath, output_file),'w') as out:
            in1 = open(self.rec_filename, "r")
            in2 = open(self.lig_filename, "r")
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

        pdbfile = os.path.join(self.mypath, output_file)
        my_struct = pdbtools.read_pdb(pdbfile)
        try:
            depth_dict = pdb_resdepth.calculate_resdepth(structure=my_struct, pdb_filename=pdbfile, method=depth)
        except:
            return
        distmat = matrice_distances.calc_distance_matrix(
            structure=my_struct,
            depth=depth_dict,
            chain_R=self.rec_chain,
            chain_L=self.lig_chain,
            dist_max=dist,
            method=depth,
        )

        vdw = Lennard_Jones.lennard_jones(dist_matrix=distmat)
        electro = electrostatic.electrostatic(inter_resid_dict=distmat, pH=pH)
        score = vdw + electro

        return score, args[1], args[2], args[3]

    def pdbpre(self, file1):
        with open(os.path.join(self.path, file1), "r") as pdb_in:   # TODO: Args.pdb correct info
            with open(file1 + "1.pdb", "w") as out: 
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
        return file1 + "1.pdb"

    def generate_one_frog(self, uid):
        Quater = [0, 0, 0, 0]
        recRandIdx = rng.integers(0, self.rec_coord.shape[0] - 1)
        ligRandIdx = rng.integers(0, self.lig_coord.shape[0] - 1)
        axis = self.rec_coord[recRandIdx]
        a = self.rec_normal[recRandIdx]
        b = self.lig_normal[ligRandIdx]

        dotProduct = np.dot(a, b)
        theta = np.arccos(dotProduct) * 2 - np.pi

        Quater = Quaternion(axis=a, angle=theta)

        final = np.array([Quater.rotate(i) for i in self.lig_atom])
        args = [[final, uid, Quater, -1]]
        return args
    
    def generate_one_frog2(self, uid):
        Quater = [0, 0, 0, 0]
        recRandIdx = rng.integers(0, self.rec_coord.shape[0] - 1)
        ligRandIdx = rng.integers(0, self.lig_coord.shape[0] - 1)
        axis = self.rec_coord[recRandIdx]
        a = self.rec_normal[recRandIdx]
        b = self.lig_normal[ligRandIdx]

        dotProduct = np.dot(a, b)
        theta = np.arccos(dotProduct) * 2 - np.pi

        Quater = Quaternion(axis=a, angle=theta)

        final = np.array([Quater.rotate(i) for i in self.lig_atom])
        args = [[final, uid, Quater, -1]]
        return args
    
    def generate_one_frog1(self, uid):
        Quater = [0, 0, 0, 0]
        recRandIdx = rng.integers(0, self.rec_coord.shape[0] - 1)
        ligRandIdx = rng.integers(0, self.lig_coord.shape[0] - 1)
        a = self.rec_coord[recRandIdx]
        b = self.lig_coord[ligRandIdx]
        tran = b - a
        dotProduct = np.dot(a, b)
        theta = np.arccos(dotProduct) * 2 - np.pi

        Quater = Quaternion(axis=a, angle=theta)
        #lig_trans = [i + tran for i in self.lig_atom]
        final = np.array([Quater.rotate(i + tran) for i in self.lig_atom])
        args = [[final, uid, Quater, -1]]
        return args

    def generate_init_population(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            Doargs = []
            for _ in range(self.frogs):
                Doargs += self.generate_one_frog(self.init)
                self.init += 1
            
            results = executor.map(self.find_score, Doargs)
            for r in results:
                if r:
                    self.structinfo[r[1]] = [r[0], r[2]]   

    def sort_frog(self):
        sorted_fitness = np.array(sorted(self.structinfo, key = lambda x: self.structinfo[x][0]))

        memeplexes = np.zeros((self.mplx_no, self.FrogsEach))

        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                memeplexes[i, j] = sorted_fitness[i + (self.mplx_no*j)] 
        return memeplexes
                
    def local_search_one_memeplex(self, im):
        """
            q: The number of frogs in submemeplex
            N: No of mutations
        """

        for iN in range(self.N):
            uId = self.init + im + 1
            rValue = rng.random(self.FrogsEach) * self.weights                      # random value with probability weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])                  # index of selected frogs in memeplex
            submemeplex = self.memeplexes[im][subindex] 

            #--- Improve the worst frog's position ---#
            # Learn from local best Pb #
            Pb = self.structinfo[int(submemeplex[0])]                               # mark the best frog in submemeplex
            Pw = self.structinfo[int(submemeplex[self.q - 1])]                      # mark the worst frog in submemeplex

            S = rng.random() * (Pb[1] - Pw[1]) 
            Uq = Pw[1] + S

            globStep = False
            censorship = False
            
            # Check feasible space and the performance #
            if self.omega[0] <= min(Uq) and max(Uq) <= self.omega[1]:
                final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                results = self.find_score([final, uId, Uq, im])

                if results[0] > Pw[0]:
                    globStep = True
            else: 
                globStep = True

            if globStep:
                S = rng.random() * (self.Frog_gb[1] - Pw[1])
                for i in range(4):
                    if S[i] > 0:
                        S[i] = min(S[i], self.max_step)
                    else:
                        S[i] = max(S[i], -self.max_step)
                Uq = Pw[1] + S

                if self.omega[0] <= min(Uq) and max(Uq) <= self.omega[1]:
                    final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                    results = self.find_score([final, uId, Uq, im])
                    if results[0] > Pw[0]:
                        censorship = True
                else:
                    censorship = True

            if censorship:
                params = self.generate_one_frog(uId)
                results = self.find_score(params)            


            #StructInfo[im] = [results[0], results[2]]
            shutil.move(os.path.join('poses/', 'out'+str(uId)+'.pdb'), os.path.join('poses/', 'out'+ str(submemeplex[self.q-1]) + '.pdb'))
            self.structinfo[int(submemeplex[self.q-1])] = [results[0], results[2]]
            self.memeplexes[im] = self.memeplexes[im][np.argsort(self.memeplexes[im])]
            
    def local_search(self):
        self.Frog_gb = self.structinfo[int(self.memeplexes[0][0])]
    
        with concurrent.futures.ProcessPoolExecutor() as executor:
            doargs = [[im] for im in range(len(self.memeplexes))]
            results = executor.map(self.local_search_one_memeplex, doargs)
    
    def shuffle_memeplexes(self):
        """Shuffles the memeplexes and sorting them.
        """
        temp = self.memeplexes.flatten()
        temp = np.array(sorted(temp, key = lambda x: self.structinfo[x][0]))
        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                self.memeplexes[i, j] = temp[i + (self.mplx_no * j)]    
            
    def run_sfla(self, data_path, protein_name, rec_name, lig_name):
        self.path = data_path
        self.rec_name = rec_name
        self.lig_name = lig_name
        
        rpdb = rec_name + '_model_st.pdb'
        lpdb = lig_name + '_model_st.pdb'
        
        self.rec_chain = [i for i in rec_name]
        self.lig_chain = [i for i in lig_name]
         
        self.rec_filename = self.pdbpre(rpdb) # INP2
        self.lig_filename = self.pdbpre(lpdb) # INP1
        
        self.rec_coord, rec_res, rec_res_id, self.rec_atom = self.chaindef(self.rec_filename, self.rec_chain)   
        self.lig_coord, lig_res, lig_res_id, self.lig_atom = self.chaindef(self.lig_filename, self.lig_chain)
        
        self.rec_normal, rec_curve = self.findPointNormals(self.rec_coord, 20,[0,0,0], rec_res_id, rec_res, 'r')
        self.lig_normal, lig_curve = self.findPointNormals(self.lig_coord, 20,[0,0,0], lig_res_id, lig_res, 'r')
        
        self.generate_init_population()
        self.memeplexes = self.sort_frog(self.mplx_no)
        
        self.omega = [np.amin(self.rec_normal), np.amax(self.rec_normal)]
        self.max_step = (self.omega[1] - self.omega[0])/2
        
        for _ in range(self.n_iter):
            self.local_search()
            self.shuffle_memeplexes()
        
        directory = "native_" + protein_name
        final_path = os.path.join("./", directory)
        os.mkdir(final_path)
        # move best global frog to native folder
        shutil.move(os.path.join("poses/", "out" + str(self.memeplexes[0][0]) + ".pdb"), directory)
    
    def run_sfla1(self, data_path, protein_name, rec_name, lig_name):
        
        self.path = data_path
        self.rec_name = rec_name
        self.lig_name = lig_name
        
        rpdb = rec_name + '_model_st.pdb'
        lpdb = lig_name + '_model_st.pdb'
        
        self.rec_chain = [i for i in rec_name]
        self.lig_chain = [i for i in lig_name]
         
        self.rec_filename = self.pdbpre(rpdb) # INP2
        self.lig_filename = self.pdbpre(lpdb) # INP1
        
        self.rec_coord, rec_res, rec_res_id, self.rec_atom = self.chaindef(self.rec_filename, self.rec_chain)   
        self.lig_coord, lig_res, lig_res_id, self.lig_atom = self.chaindef(self.lig_filename, self.lig_chain)
        
        self.generate_init_population()