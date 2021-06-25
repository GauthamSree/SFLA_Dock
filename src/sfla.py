import os, glob, sys, logging, sys
import concurrent.futures
import argparse

import shutil

import numpy as np
from Bio.PDB import *
from pyquaternion import Quaternion

from utils.Complex import Complex
from utils import pdbtools
from utils import pdb_resdepth
from utils import matrice_distances
from utils import Lennard_Jones
from utils import electrostatic
from utils import poses

logging.basicConfig(
    format="%(asctime)s [%(name)s: %(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log").setLevel(logging.DEBUG),
        logging.StreamHandler(sys.stdout).setLevel(logging.INFO)
    ]
)
logger = logging.getLogger(__name__)

MAX_TRANSITION = 15

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
        logger.info(f"out{args[1]}.pdb -- Calculating score")
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
        logger.info(f"out{args[1]}.pdb -- score {score:.3f}")
        return score, args[1], args[2], args[3]

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

    def generate_one_frog(self, uid, initial=False):
        rec_rand_idx = rng.integers(0, self.receptor.coord.shape[0] - 1)
        lig_rand_idx = rng.integers(0, self.ligand.coord.shape[0] - 1)
        a = self.receptor.coord[rec_rand_idx]
        b = self.ligand.coord[lig_rand_idx]
        if initial:
            trans_coord = self.rand_trans_coord[uid]
        else:
            trans_rand_idx = rng.integers(self.frogs, self.rand_trans_coord.shape[0] - 1)
            trans_coord = self.rand_trans_coord[trans_rand_idx]
        
        quater = self.quaternion_from_vectors(a, b)
        final = self.ligand.tranformation(quater, trans_coord)
        args = [[final, uid, quater, trans_coord]]
        return args

    def generate_init_population(self):
        logger.info(f"Generating initial population (Number of frogs: {self.frogs})")
        self.rand_trans_coord, self.receptor_max_diameter, self.ligand_max_diameter = poses.calculate_initial_poses(self.receptor, 
                                                                                                                    self.ligand, 
                                                                                                                    self.frogs*2)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            Doargs = []
            for _ in range(self.frogs):
                Doargs += self.generate_one_frog(self.init, initial=True)
                self.init += 1
            
            results = executor.map(self.find_score, Doargs)
            for r in results:
                if r:
                    self.structinfo[r[1]] = [r[0], r[2], r[3]]

    def sort_frog(self):
        logger.info(f"Sorting the frogs and making {self.mplx_no} memeplexes with {self.frogs} frogs each")
        sorted_fitness = np.array(sorted(self.structinfo, key = lambda x: self.structinfo[x][0]))

        memeplexes = np.zeros((self.mplx_no, self.FrogsEach))

        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                memeplexes[i, j] = sorted_fitness[i + (self.mplx_no*j)] 
        return memeplexes

    def clip_the_point_max_step(self, worst_frog_coord, best_frog_coord):
        t = self.step_size/np.linalg.norm(worst_frog_coord - best_frog_coord)
        new_coord = (1 - t) * worst_frog_coord + (t * best_frog_coord)
        return new_coord

    def new_step(self, quart_best, tran_coord_best, quart_worst, tran_coord_worst):
        quart_new = Quaternion.slerp(quart_worst, quart_best, rng.random())
        shift_coord = (tran_coord_best - tran_coord_worst) * rng.random()
        
        if np.linalg.norm(tran_coord_worst - shift_coord) >= self.step_size:
            trans_coord = self.clip_the_point_max_step(tran_coord_worst, tran_coord_best)
        else:
            trans_coord = tran_coord_worst + shift_coord
        
        return quart_new, trans_coord

    def local_search_one_memeplex(self, im):
        """
            q: The number of frogs in submemeplex
            N: No of mutations
        """
        for idx in range(self.N):
            logger.info(f"Local search of Memeplex {im + 1}: Mutation {idx}/{self.N}")
            uId = self.init + im + 1
            rValue = rng.random(self.FrogsEach) * self.weights                      # random value with probability weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])                  # index of selected frogs in memeplex
            submemeplex = self.memeplexes[im][subindex] 

            #--- Improve the worst frog's position ---#
            # Learn from local best Pb #
            Pb = self.structinfo[int(submemeplex[0])]                               # mark the best frog in submemeplex
            Pw = self.structinfo[int(submemeplex[self.q - 1])]                      # mark the worst frog in submemeplex
            quart_new, trans_coord = self.new_step(Pb[1], Pb[2], Pw[1], Pw[2])

            globStep = False
            censorship = False
            
            # Check feasible space and the performance #
            if np.linalg.norm(trans_coord) <= self.omega:
                logger.info(f"out{uId}.pdb: Learn from local best Pb")
                final = self.ligand.tranformation(quart_new, trans_coord)   #final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                results = self.find_score([final, uId, quart_new, trans_coord])

                if results[0] > Pw[0]:
                    globStep = True
            else: 
                globStep = True

            if globStep:
                logger.info(f"out{uId}.pdb: Learn from global best Pb since score didn't improve")
                quart_new, trans_coord = self.new_step(self.Frog_gb[1], self.Frog_gb[2], Pw[1], Pw[2])
              
                if np.linalg.norm(trans_coord) <= self.omega:
                    final = self.ligand.tranformation(quart_new, trans_coord)   #final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                    results = self.find_score([final, uId, quart_new, trans_coord])
                    if results[0] > Pw[0]:
                        censorship = True
                else:
                    censorship = True

            if censorship:
                logger.info(f"out{uId}.pdb: generating a new frog since score didn't improve")
                params = self.generate_one_frog(uId)
                results = self.find_score(params)            


            shutil.move(os.path.join('poses/', 'out'+str(uId)+'.pdb'), os.path.join('poses/', 'out'+ str(submemeplex[self.q-1]) + '.pdb'))
            self.structinfo[int(submemeplex[self.q-1])] = [results[0], results[2], results[3]]
            self.memeplexes[im] = self.memeplexes[im][np.argsort(self.memeplexes[im])]
            logger.info(f"Local search of Memeplex {im + 1}: Mutation {idx}/{self.N} finished")

         
    def local_search(self):
        self.Frog_gb = self.structinfo[int(self.memeplexes[0][0])]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            doargs = [im for im, _ in enumerate(self.memeplexes)]
            results = executor.map(self.local_search_one_memeplex, doargs)

    def shuffle_memeplexes(self):
        """Shuffles the memeplexes and sorting them.
        """
        logger.info("Shuffling the memeplexes and sorting them")
        temp = self.memeplexes.flatten()
        temp = np.array(sorted(temp, key = lambda x: self.structinfo[x][0]))
        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                self.memeplexes[i, j] = temp[i + (self.mplx_no * j)]    
            
    def run_sfla(self, data_path, protein_name, rec_name, lig_name):
        logger.info("Starting SFLA algorithm")
        self.path = data_path
        self.receptor = Complex(rec_name, data_path)
        self.ligand = Complex(lig_name, data_path)
        
        self.generate_init_population()
        self.memeplexes = self.sort_frog()
        
        self.omega = 2 * self.receptor_max_diameter  # TODO: Two times or One time
        self.step_size = self.ligand_max_diameter / 4

        for idx in range(self.n_iter):
            logger.info(f"Local Search: {idx}/{self.n_iter}")
            self.local_search()
            self.shuffle_memeplexes()
        
        directory = "native_" + protein_name
        final_path = os.path.join("./", directory)
        logger.info(f"Creating a new directory - {final_path}")
        os.mkdir(final_path)
        # move best global frog to native folder
        logger.info(f"Moving global best frog to the new directory - {final_path}")
        shutil.move(os.path.join("poses/", "out" + str(self.memeplexes[0][0]) + ".pdb"), directory)
    
    def run_sfla_test(self, data_path, protein_name, rec_name, lig_name):
        logger.info("Starting SFLA algorithm")
        self.path = data_path
        self.receptor = Complex(rec_name, data_path)
        self.ligand = Complex(lig_name, data_path)
        
        self.generate_init_population()
        self.memeplexes = self.sort_frog()
        
        self.omega = 2 * self.receptor_max_diameter  # TODO: Two times or One time
        self.step_size = self.ligand_max_diameter / 4

    def run_remaining(self):
        for _ in range(2):
            logger.info(f"Local Search: {idx}/{self.n_iter}")
            self.local_search()
            self.shuffle_memeplexes()
        