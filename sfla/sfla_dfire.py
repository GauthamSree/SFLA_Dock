from enum import unique
import os, sys, logging
import argparse, shutil
import concurrent.futures

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
from utils import dfire


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s: %(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

rng = np.random.default_rng(0)


class SFLA:
    def __init__(self, frogs, mplx_no, no_of_iteration, no_of_mutation, q):
        self.frogs = frogs
        self.mplx_no = mplx_no
        self.FrogsEach = int(self.frogs/self.mplx_no)
        self.weights = [2*(self.FrogsEach+1-j)/(self.FrogsEach*(self.FrogsEach+1)) for j in range(1, self.FrogsEach+1)] 
        self.conformation_data = {}
        self.mypath ='poses/'
        if not os.path.exists(self.mypath):
            os.mkdir(os.path.join("./", self.mypath))
        self.no_of_iteration = no_of_iteration
        self.no_of_mutation = no_of_mutation
        self.q = q
    
    def __repr__(self):
        return f"SFLA (Frogs = {self.frogs}, Memeplexes = {self.mplx_no})"
    
    def __str__(self):
        return f"SFLA (Frogs = {self.frogs}, Memeplexes = {self.mplx_no})"

    @property
    def memeplexes(self):
        return self._memeplexes

    @memeplexes.setter
    def memeplexes(self, memeplexes):
        self._memeplexes = memeplexes

    def find_score(self, args):
        #logger.info(f"out{args[1]}.pdb: Calculating score")
        output_file = self.write_to_file(args[0], args[1])
        pdbfile = os.path.join(self.mypath, output_file)
        my_struct = pdbtools.read_pdb(pdbfile)
        try:
            depth_dict = pdb_resdepth.calculate_resdepth(structure=my_struct, pdb_filename=pdbfile, method="msms")
        except:
            return
        
        distmat = matrice_distances.calc_distance_matrix(
            structure=my_struct,
            depth=depth_dict,
            chain_R=self.receptor.chain,
            chain_L=self.ligand.chain,
            dist_max=8.6,
            method="msms",
        )

        vdw = Lennard_Jones.lennard_jones(dist_matrix=distmat)
        electro = electrostatic.electrostatic(inter_resid_dict=distmat, pH=7)
        score = vdw + electro
        logger.info(f"out{args[1]}.pdb -- score = {score:.3f}")
        return score, args[1], args[2], args[3]

    def find_score_dfire(self, args):
        #logger.info(f"out{args[1]}.pdb: Calculating score")
        output_file = self.write_to_file(args[0], args[1])
        dfire_score = self.dfire_model(self.receptor, self.receptor.atom_coord, self.ligand, args[0])
        logger.info(f"out{args[1]}.pdb -- score = {dfire_score:.3f}")
        return dfire_score, args[1], args[2], args[3]

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

    def generate_one_frog(self, unique_id, initial=False):
        rec_rand_idx = rng.integers(0, self.receptor.coord.shape[0] - 1)
        lig_rand_idx = rng.integers(0, self.ligand.coord.shape[0] - 1)
        a = self.receptor.coord[rec_rand_idx]
        b = self.ligand.coord[lig_rand_idx]
        if initial:
            trans_coord = self.rand_trans_coord[unique_id]
        else:
            trans_coord = poses.generate_new_pose(a, self.ligand_max_diameter, rng)
        
        quater = self.quaternion_from_vectors(a, b)
        final = self.ligand.tranformation(quater, trans_coord)
        args = [final, unique_id, quater, trans_coord]
        return args
    
    def generate_init_population(self):
        logger.info(f"Generating initial population (Number of frogs: {self.frogs})")
        (
            self.rand_trans_coord,
            self.receptor_max_diameter,
            self.ligand_max_diameter,
        ) = poses.calculate_initial_poses(self.receptor, self.ligand, self.frogs)
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            score_args = [self.generate_one_frog(frog_id, initial=True) for frog_id in range(self.frogs)]
            results = executor.map(self.find_score, score_args)
            for r in results:
                if r:
                    self.conformation_data[r[1]] = [r[0], r[2], r[3]]

    def generate_init_population_dfire(self):
        logger.info(f"Generating initial population (Number of frogs: {self.frogs})")
        (
            self.rand_trans_coord,
            self.receptor_max_diameter,
            self.ligand_max_diameter,
        ) = poses.calculate_initial_poses(self.receptor, self.ligand, self.frogs)
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            score_args = [self.generate_one_frog(frog_id, initial=True) for frog_id in range(self.frogs)]
            results = executor.map(self.find_score_dfire, score_args)
            for r in results:
                if r:
                    self.conformation_data[r[1]] = [r[0], r[2], r[3]]

    def sort_frog(self):
        logger.info(f"Sorting the frogs and making {self.mplx_no} memeplexes with {self.frogs} frogs each")
        sorted_fitness = np.array(sorted(self.conformation_data, key = lambda x: self.conformation_data[x][0], reverse=True))

        memeplexes = np.zeros((self.mplx_no, self.FrogsEach))

        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                memeplexes[i, j] = sorted_fitness[i + (self.mplx_no*j)] 
        return memeplexes

    def clip_the_point_max_step(self, worst_frog_coord, best_frog_coord):
        norm = np.linalg.norm(worst_frog_coord - best_frog_coord)
        if not np.isnan(norm) and norm < 0.00001:
            t = 0.9
        else: 
            t = self.step_size/norm
        new_coord = (1 - t) * worst_frog_coord + (t * best_frog_coord)
        return new_coord

    def new_step(self, quart_best, tran_coord_best, quart_worst, tran_coord_worst):
        quart_new = Quaternion.slerp(quart_worst, quart_best, rng.random())
        shift_coord = (tran_coord_best - tran_coord_worst) * rng.random()
        
        if np.linalg.norm(tran_coord_worst - shift_coord) > self.step_size:
            trans_coord = self.clip_the_point_max_step(tran_coord_worst, tran_coord_best)
        else:
            trans_coord = tran_coord_worst + shift_coord
        
        return quart_new, trans_coord

    def local_search_one_memeplex_dfire(self, args):
        """
            im: memeplex_idx
        """
        im, iter_idx = args 
        memplex = self.memeplexes[im]
        extracted_conformation_data = {int(item):self.conformation_data.get(item) for item in memplex}
        
        for idx in range(self.no_of_mutation):
            logger.info(f"Iteration {iter_idx} -- Local search of Memeplex {im + 1}: Mutation {idx + 1}/{self.no_of_mutation}")
            unique_id = self.frogs + im + 1
            rValue = rng.random(self.FrogsEach) * self.weights                      # random value with probability weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])                  # index of selected frogs in memeplex
            submemeplex = memplex[subindex] 

            #--- Improve the worst frog's position ---#
            # Learn from local best Pb #
            Pb = extracted_conformation_data[int(submemeplex[0])]                               # mark the best frog in submemeplex
            Pw = extracted_conformation_data[int(submemeplex[self.q - 1])]                      # mark the worst frog in submemeplex
            quart_new, trans_coord = self.new_step(Pb[1], Pb[2], Pw[1], Pw[2])

            globStep = False
            censorship = False
            
            # Check feasible space and the performance #
            if np.linalg.norm(self.receptor.COM - trans_coord) <= self.omega:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): Learn from local best Pb")
                final = self.ligand.tranformation(quart_new, trans_coord)   #final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                results = self.find_score_dfire([final, unique_id, quart_new, trans_coord])

                if results[0] < Pw[0]:
                    globStep = True
            else: 
                globStep = True

            if globStep:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): score didn't improve... Learn from global best Pb")
                quart_new, trans_coord = self.new_step(self.Frog_gb[1], self.Frog_gb[2], Pw[1], Pw[2])
              
                if np.linalg.norm(self.receptor.COM - trans_coord) <= self.omega:
                    final = self.ligand.tranformation(quart_new, trans_coord)   #final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                    results = self.find_score_dfire([final, unique_id, quart_new, trans_coord])
                    if results[0] < Pw[0]:
                        censorship = True
                else:
                    censorship = True

            if censorship:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): score didn't improve... generating a new frog")
                params = self.generate_one_frog(unique_id)
                results = self.find_score_dfire(params)            

            shutil.move(os.path.join('poses/', 'out'+str(int(unique_id))+'.pdb'), os.path.join('poses/', 'out'+ str(int(submemeplex[self.q-1])) + '.pdb'))
            extracted_conformation_data[int(submemeplex[self.q-1])] = [results[0], results[2], results[3]]
            memplex = np.array(sorted(extracted_conformation_data, key = lambda x: extracted_conformation_data[x][0], reverse=True))
            logger.info(f"Iteration {iter_idx} -- Local search of Memeplex {im + 1}: pose moved to out{int(submemeplex[self.q-1])}.pdb ::: Mutation {idx + 1}/{self.no_of_mutation} finished")

        return (extracted_conformation_data, im, memplex)

    def local_search_one_memeplex(self, args):
        """
            im: memeplex_idx
        """
        im, iter_idx = args 
        memplex = self.memeplexes[im]
        extracted_conformation_data = {int(item):self.conformation_data.get(item) for item in memplex}
        
        for idx in range(self.no_of_mutation):
            logger.info(f"Iteration {iter_idx} -- Local search of Memeplex {im + 1}: Mutation {idx + 1}/{self.no_of_mutation}")
            unique_id = self.frogs + im + 1
            rValue = rng.random(self.FrogsEach) * self.weights                      # random value with probability weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])                  # index of selected frogs in memeplex
            submemeplex = memplex[subindex] 

            #--- Improve the worst frog's position ---#
            # Learn from local best Pb #
            Pb = extracted_conformation_data[int(submemeplex[0])]                               # mark the best frog in submemeplex
            Pw = extracted_conformation_data[int(submemeplex[self.q - 1])]                      # mark the worst frog in submemeplex
            quart_new, trans_coord = self.new_step(Pb[1], Pb[2], Pw[1], Pw[2])

            globStep = False
            censorship = False
            
            # Check feasible space and the performance #
            if np.linalg.norm(self.receptor.COM - trans_coord) <= self.omega:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): Learn from local best Pb")
                final = self.ligand.tranformation(quart_new, trans_coord)   #final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                results = self.find_score([final, unique_id, quart_new, trans_coord])

                if results[0] > Pw[0] and results[0] > 0:
                    globStep = True
            else: 
                globStep = True

            if globStep:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): score didn't improve... Learn from global best Pb")
                quart_new, trans_coord = self.new_step(self.Frog_gb[1], self.Frog_gb[2], Pw[1], Pw[2])
              
                if np.linalg.norm(self.receptor.COM - trans_coord) <= self.omega:
                    final = self.ligand.tranformation(quart_new, trans_coord)   #final = np.array([Uq.rotate(i) for i in self.lig_atom])  
                    results = self.find_score([final, unique_id, quart_new, trans_coord])
                    if results[0] > Pw[0] and results[0] > 0:
                        censorship = True
                else:
                    censorship = True

            if censorship:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): score didn't improve... generating a new frog")
                params = self.generate_one_frog(unique_id)
                results = self.find_score(params)            

            shutil.move(os.path.join('poses/', 'out'+str(int(unique_id))+'.pdb'), os.path.join('poses/', 'out'+ str(int(submemeplex[self.q-1])) + '.pdb'))
            extracted_conformation_data[int(submemeplex[self.q-1])] = [results[0], results[2], results[3]]
            memplex = np.array(sorted(extracted_conformation_data, key = lambda x: extracted_conformation_data[x][0]))
            logger.info(f"Iteration {iter_idx} -- Local search of Memeplex {im + 1}: pose moved to out{int(submemeplex[self.q-1])}.pdb ::: Mutation {idx + 1}/{self.no_of_mutation} finished")

        return (extracted_conformation_data, im, memplex)
    
    def local_search(self, iter_idx):
        self.Frog_gb = self.conformation_data[int(self.memeplexes[0][0])]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            doargs = [[im, iter_idx] for im, _ in enumerate(self.memeplexes)]
            results = executor.map(self.local_search_one_memeplex, doargs)
            
            for r in results:
                if r:
                    self.conformation_data.update(r[0])
                    self.memeplexes[r[1]] = r[2]

    def local_search_dfire(self, iter_idx):
        self.Frog_gb = self.conformation_data[int(self.memeplexes[0][0])]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            doargs = [[im, iter_idx] for im, _ in enumerate(self.memeplexes)]
            results = executor.map(self.local_search_one_memeplex_dfire, doargs)
            
            for r in results:
                if r:
                    self.conformation_data.update(r[0])
                    self.memeplexes[r[1]] = r[2]
    
    def shuffle_memeplexes(self):
        """Shuffles the memeplexes and sorting them.
        """
        logger.info("Shuffling the memeplexes and sorting them")
        temp = self.memeplexes.flatten()
        temp = np.array(sorted(temp, key = lambda x: self.conformation_data[x][0], reverse=True))
        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                self.memeplexes[i, j] = temp[i + (self.mplx_no * j)]    
            
    def run_sfla(self, data_path, protein_name, rec_name, lig_name):
        logger.info("Starting SFLA algorithm")
        self.path = data_path
        self.receptor = Complex(rec_name, data_path)
        self.ligand = Complex(lig_name, data_path)
        self.ligand.move_to_origin(inplace=True)
        
        self.receptor.dfire_objects = dfire.get_dfire_objects(self.receptor.structure)
        self.ligand.dfire_objects = dfire.get_dfire_objects(self.ligand.structure)
        
        self.dfire_model = dfire.DFIRE()

        # self.generate_init_population()
        self.generate_init_population_dfire()
        self.memeplexes = self.sort_frog()
        
        self.omega = 1.5 * self.receptor_max_diameter
        self.step_size = 1

        for idx in range(self.no_of_iteration):
            logger.info(f"Local Search: {idx+1}/{self.no_of_iteration}")
            # self.local_search(idx)
            self.local_search_dfire(idx)
            self.shuffle_memeplexes()

        directory = "native_" + protein_name
        final_path = os.path.join("./", directory)
        logger.info(f"Creating a new directory - {final_path}")
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        # move best global frog to native folder
        logger.info(f"Moving best frog from each memeplexes to the new directory - {final_path} and saving the best energy.")
        with open("best_energy.txt", 'w') as best_eng:                
            for im, memeplex in enumerate(self.memeplexes):
                unique_id = int(memeplex[0])
                best_eng.write(f"Memeplex {im+1} (out{str(unique_id)}.pdb) --- Score: {self.conformation_data[unique_id][0]}\n")
                shutil.move(os.path.join("poses/", "out" + str(unique_id) + ".pdb"), directory)
        
        with open("all_frog_energy.txt", 'w') as all_frog:
            for key, value in self.conformation_data.items():
                all_frog.write(f"out{key}.pdb --- Score = {value[0]}\n")
                
    
    def run_sfla_test(self, data_path, protein_name, rec_name, lig_name):
        logger.info("Starting SFLA algorithm")
        self.path = data_path
        self.receptor = Complex(rec_name, data_path)
        self.ligand = Complex(lig_name, data_path)
        self.ligand.move_to_origin(inplace=True)
        
        self.receptor.dfire_objects = dfire.get_dfire_objects(self.receptor.structure)
        self.ligand.dfire_objects = dfire.get_dfire_objects(self.ligand.structure)
        
        self.dfire_model = dfire.DFIRE()

        # self.generate_init_population()
        self.generate_init_population_dfire()
        self.memeplexes = self.sort_frog()
        
        self.omega = 1.5 * self.receptor_max_diameter  # TODO: Two times or One time
        self.step_size = 1

    def run_remaining(self):
        for idx in range(5):
            logger.info(f"Local Search: {idx+1}/{5}")
            self.local_search_dfire(idx)
            self.shuffle_memeplexes()

    
if __name__ == "__main__":
    #"Data/4dn4_LH:M"

    parser = argparse.ArgumentParser(description='Shuffled Frog Leap Algorithm (SFLA)')
    parser.add_argument('-p', '--pdb', type=str, required=True, help='PDB File Directory')
    parser.add_argument('-n', type=int, required=True, help='Number of Iterations')
    args = parser.parse_args()
    
    n = args.n
    protein_name = args.pdb.split('/')[-1].split('_')[0]
    rec_lig_name = args.pdb.split('/')[-1].split('_')[1].split(':')

    sfla = SFLA(frogs=400, mplx_no=40, no_of_iteration=n, no_of_mutation=10, q=6)  # TODO: 400, 40, n, 10, 6 
    #sfla = SFLA(frogs=50, mplx_no=10, no_of_iteration=n, no_of_mutation=2, q=4)
    sfla.run_sfla(str(args.pdb), protein_name, rec_lig_name[0], rec_lig_name[1])