import os, glob, sys, math
import concurrent.futures
import argparse

import shutil

import numpy as np
import pandas as pd
from Bio.PDB import *
from pyquaternion import Quaternion

from utils import Complex
from utils import pdbtools
from utils import pdb_resdepth
from utils import matrice_distances
from utils import Lennard_Jones
from utils import electrostatic

MAX_TRANSITION = 30

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
    
    def __repr__(self):
        return f"SFLA (Frogs = {self.frogs}, Memeplexes = {self.mplx_no})"
    
    def __str__(self):
        return f"SFLA (Frogs = {self.frogs}, Memeplexes = {self.mplx_no})"
    
    def find_score(self, args):
        output_file = self.write_to_file(args[0], args[1])
        pH = 7
        dist = 8.6
        depth = "msms"
        pdbfile = os.path.join(self.mypath, output_file)
        my_struct = pdbtools.read_pdb(pdbfile)
        try:
            depth_dict = pdb_resdepth.calculate_resdepth(structure=my_struct, pdb_filename=pdbfile, method=depth)
        except:
            return
        
        distmat = matrice_distances.calc_distance_matrix(
            structure=my_struct,
            depth=depth_dict,
            chain_R=self.receptor.chain,
            chain_L=self.ligand.chain,
            dist_max=dist,
            method=depth,
        )

        vdw = Lennard_Jones.lennard_jones(dist_matrix=distmat)
        electro = electrostatic.electrostatic(inter_resid_dict=distmat, pH=pH)
        score = vdw + electro

        return score, args[1], args[2], args[3], args[4]

    def write_to_file(self, new_coord, id):
        output_file = 'out' + str(id) + '.pdb'
        with open(os.path.join(self.mypath, output_file),'w') as out:
            in1 = open(self.receptor.pdb_file, "r")
            in2 = open(self.ligand.pdb_file, "r")
            for line in in1:
                if "ATOM" in line:
                    out.write(line)
            for indexing, line in enumerate(in2):
                if "ATOM" in line:
                    l = line.split()
                    l[0] = l[0].ljust(5)
                    l[1] = l[1].rjust(5)
                    l[2] = l[2].ljust(3)
                    l[3] = l[3].ljust(3)
                    l[4] = line[21]
                    l[5] = ("%4d" % (int(line[22:26]))).rjust(4)
                    l[6] = ("%8.3f" % (float(new_coord[indexing][0]))).rjust(8)
                    l[7] = ("%8.3f" % (float(new_coord[indexing][1]))).rjust(8)
                    l[8] = ("%8.3f" % (float(new_coord[indexing][2]))).rjust(8)
                    out.write(
                        "{0} {1}  {2} {3} {4}{5}    {6}{7}{8}".format(
                            l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]
                        )
                    )
                    out.write("\n")
        return output_file

    def normalize_vector(self, v):
        """Normalizes a given vector"""
        norm = np.linalg.norm(v)
        if norm < 0.00001:
            return v
        return v / norm
    
    def quaternion_from_vectors(self, a, b):
        """Calculate quaternion between two vectors a and b.
        """
        u = self.normalize_vector(a)
        v = self.normalize_vector(b)
        w = np.cross(u, v)
        q = Quaternion(np.dot(u, v), w[0], w[1], w[2])
        q[0] += q.magnitude
        
        return q.normalised

    def generate_one_frog_test(self, uid):
        recRandIdx = rng.integers(0, self.receptor.coord.shape[0] - 1)
        ligRandIdx = rng.integers(0, self.ligand.coord.shape[0] - 1)
        
        a = self.receptor.coord[recRandIdx]
        b = self.ligand.coord[ligRandIdx]
        trans_coord = rng.uniform(-20, 20, 3)
        quater = self.quaternion_from_vectors(a, b)
        
        final = self.ligand.tranformation(quater, trans_coord)
        args = [[final, uid, quater, trans_coord, -1]]
        return args
    
    def generate_one_frog(self, uid):
        recRandIdx = rng.integers(0, self.receptor.coord.shape[0] - 1)
        ligRandIdx = rng.integers(0, self.ligand.coord.shape[0] - 1)
        
        a = self.receptor.coord[recRandIdx]
        b = self.ligand.coord[ligRandIdx]
        trans_coord = rng.uniform(-20, 20, 3)
        quater = self.quaternion_from_vectors(a, b)
        
        final = self.ligand.tranformation(quater, trans_coord)
        args = [[final, uid, quater, trans_coord, -1]]
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
                    self.structinfo[r[1]] = [r[0], r[2], r[3]]   

    def sort_frog(self):
        sorted_fitness = np.array(sorted(self.structinfo, key = lambda x: self.structinfo[x][0]))

        memeplexes = np.zeros((self.mplx_no, self.FrogsEach))

        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                memeplexes[i, j] = sorted_fitness[i + (self.mplx_no*j)] 
        return memeplexes

    def get_minmax_crd(self, times=2):
        coord = (np.max(self.receptor.coord, axis=0) - np.min(self.receptor.coord, axis=0)) * times 
        return coord

    def new_step(self, q1, t1, q2, t2):
        q_new = Quaternion.slerp(q1, q2, rng.random())
        coord_new = np.clip((t2 - t1) * rng.random(), -MAX_TRANSITION, MAX_TRANSITION)
        t_coord = t2 + coord_new
        return q_new, t_coord 

    def local_search_one_memeplex(self, im):
        """
            q: The number of frogs in submemeplex
            N: No of mutations
        """

        for _ in range(self.N):
            uId = self.init + im + 1
            rValue = rng.random(self.FrogsEach) * self.weights                      # random value with probability weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])                  # index of selected frogs in memeplex
            submemeplex = self.memeplexes[im][subindex] 

            #--- Improve the worst frog's position ---#
            # Learn from local best Pb #
            Pb = self.structinfo[int(submemeplex[0])]                               # mark the best frog in submemeplex
            Pw = self.structinfo[int(submemeplex[self.q - 1])]                      # mark the worst frog in submemeplex

            q_new, coord_new = self.new_step(Pb[1], Pb[2], Pw[1], Pw[2])

            globStep = False
            censorship = False
            
            # Check feasible space and the performance #
            if -np.max(self.omega) <= np.min(coord_new) and np.max(coord_new) <= np.max(self.omega):
                final = self.ligand.tranformation(q_new, coord_new)   #final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                results = self.find_score([final, uId, q_new, coord_new, im])

                if results[0] > Pw[0]:
                    globStep = True
            else: 
                globStep = True

            if globStep:
                q_new, coord_new = self.new_step(self.Frog_gb[1], self.Frog_gb[2], Pw[1], Pw[2])
              
                if -np.max(self.omega) <= np.min(coord_new) and np.max(coord_new) <= np.max(self.omega):
                    final = self.ligand.tranformation(q_new, coord_new)   #final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                    results = self.find_score([final, uId, q_new, coord_new, im])
                    if results[0] > Pw[0]:
                        censorship = True
                else:
                    censorship = True

            if censorship:
                params = self.generate_one_frog(uId)
                results = self.find_score(params)            


            shutil.move(os.path.join('poses/', 'out'+str(uId)+'.pdb'), os.path.join('poses/', 'out'+ str(submemeplex[self.q-1]) + '.pdb'))
            self.structinfo[int(submemeplex[self.q-1])] = [results[0], results[2], results[3]]
            self.memeplexes[im] = self.memeplexes[im][np.argsort(self.memeplexes[im])]
            
    def local_search(self):
        self.Frog_gb = self.structinfo[int(self.memeplexes[0][0])]
    
        with concurrent.futures.ProcessPoolExecutor() as executor:
            doargs = [[im] for im in range(len(self.memeplexes))]
            _ = executor.map(self.local_search_one_memeplex, doargs)
    
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
        self.receptor = Complex(rec_name, data_path)
        self.ligand = Complex(lig_name, data_path)
        
        self.generate_init_population()
        self.memeplexes = self.sort_frog(self.mplx_no)
        
        self.omega = self.get_minmax_crd()

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
        self.receptor = Complex(rec_name, data_path)
        self.ligand = Complex(lig_name, data_path)
        
        self.generate_init_population()