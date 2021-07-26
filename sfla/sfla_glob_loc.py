import os, sys, logging
import argparse, shutil
import concurrent.futures

import numpy as np
from Bio.PDB import *
from pyquaternion import Quaternion

from utils.Complex import Complex
from utils import poses
from utils import dfire
from utils import anm

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
    def __init__(self, frogs, mplx_no, no_of_iteration, no_of_mutation, q, use_anm=True):
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
        self.use_anm = use_anm
        self.step_nmodes = anm.DEFAULT_NMODES_STEP
    
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
        output_file = self.write_to_file(args[0], args[3])
        dfire_score = self.dfire_model(self.receptor, args[1], self.ligand, args[2])
        logger.info(f"{output_file} -- score = {dfire_score:.3f}")
        return dfire_score, args[0], args[4], args[5], args[6], args[7]

    def write_to_file(self, id:int, new_coord:np.ndarray):
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

    def transform_complex(self, unique_id, quater, trans_coord, rec_nm, lig_nm):
        receptor_coordinates = self.receptor.atom_coord
        if self.use_anm:
            receptor_coordinates = self.receptor.tranformation(do_rot_trans=False, do_anm=True, anm_extent=rec_nm)
            ligand_coordinates = self.ligand.tranformation(quater, trans_coord, do_anm=True, anm_extent=lig_nm)

        ligand_coordinates_without_anm = self.ligand.tranformation(quater, trans_coord)
        ligand_coordinates = ligand_coordinates if self.use_anm else ligand_coordinates_without_anm 

        args = [unique_id, receptor_coordinates, ligand_coordinates, ligand_coordinates_without_anm, quater, trans_coord, rec_nm, lig_nm]
        return args


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
    
    # def new_quaternion_change(self, q):
    #     a = q[0] * rng.random()
    #     new_q = Quaternion(a, q[1], q[2], q[3])
    #     return new_q.normalised

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
        rec_nm, lig_nm = [np.array([])] * 2
        receptor_coordinates = self.receptor.atom_coord
        
        # ANM      
        if self.use_anm:    
            if self.receptor.num_nmodes > 0:
                rec_nm = rng.normal(anm.DEFAULT_EXTENT_MU, anm.DEFAULT_EXTENT_SIGMA, self.receptor.num_nmodes)
                rec_nm = np.clip(rec_nm, anm.MIN_EXTENT, anm.MAX_EXTENT)
                receptor_coordinates = self.receptor.tranformation(do_rot_trans=False, do_anm=True, anm_extent=rec_nm)
            
            if self.ligand.num_nmodes > 0:
                lig_nm = rng.normal(anm.DEFAULT_EXTENT_MU, anm.DEFAULT_EXTENT_SIGMA, self.ligand.num_nmodes)
                lig_nm = np.clip(lig_nm, anm.MIN_EXTENT, anm.MAX_EXTENT)
                ligand_coordinates = self.ligand.tranformation(quater, trans_coord, do_anm=True, anm_extent=lig_nm)
        
        ligand_coordinates_without_anm = self.ligand.tranformation(quater, trans_coord)
        ligand_coordinates = ligand_coordinates if self.use_anm else ligand_coordinates_without_anm
        args = [unique_id, receptor_coordinates, ligand_coordinates, ligand_coordinates_without_anm, quater, trans_coord, rec_nm, lig_nm]
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
                    self.conformation_data[r[1]] = [r[0], r[2], r[3], r[4], r[5]]

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

    def new_step(self, quart_best, tran_coord_best, rec_extent_best, lig_extent_best, quart_worst, tran_coord_worst, rec_extent_worst, lig_extent_worst):
        quart_new = Quaternion.slerp(quart_worst, quart_best, rng.random())
        shift_coord = (tran_coord_best - tran_coord_worst) * rng.random()
        
        if np.linalg.norm(tran_coord_worst - shift_coord) > self.step_size:
            trans_coord = self.clip_the_point_max_step(tran_coord_worst, tran_coord_best)
        else:
            trans_coord = tran_coord_worst + shift_coord
        
        new_rec_extent = rec_extent_worst
        new_lig_extent = lig_extent_worst

        if self.receptor.num_nmodes > 0:
            delta_x = (rec_extent_best - rec_extent_worst) * rng.random()
            n = np.linalg.norm(delta_x)
            if not np.allclose([0.0], [n]):
                delta_x *= (self.step_nmodes / n)
                new_rec_extent += delta_x
                new_rec_extent = np.clip(new_rec_extent, anm.MIN_EXTENT, anm.MAX_EXTENT)
        
        if self.ligand.num_nmodes > 0:
            delta_x = (lig_extent_best - lig_extent_worst) * rng.random()
            n = np.linalg.norm(delta_x)

            if not np.allclose([0.0], [n]):
                delta_x *= (self.step_nmodes / n)
                new_lig_extent += delta_x
                new_lig_extent = np.clip(new_lig_extent, anm.MIN_EXTENT, anm.MAX_EXTENT)
        
        return (quart_new, trans_coord, new_rec_extent, new_lig_extent)

    def new_step_test(self, local_best, local_worst):
        w = 0.72
        c1, c2 = 1.49, 1.49
        quart_gbest, tran_coord_gbest, rec_extent_gbest, lig_extent_gbest = self.Frog_gb[1], self.Frog_gb[2], self.Frog_gb[3], self.Frog_gb[4]
        quart_best, tran_coord_best, rec_extent_best, lig_extent_best = local_best[1], local_best[2], local_best[3], local_best[4]
        quart_worst, tran_coord_worst, rec_extent_worst, lig_extent_worst = local_worst[1], local_worst[2], local_worst[3], local_worst[4]
        
        # quart_new = Quaternion.slerp(quart_worst, quart_best, rng.random())
        b_quart = (c1*rng.random()*(quart_best - quart_worst) + c2*rng.random()*(quart_gbest - quart_worst)).normalised
        quart_new = Quaternion.slerp(quart_worst, b_quart, w)
        
        shift_coord = c1*rng.random()*(tran_coord_gbest - tran_coord_worst) + c2*rng.random()*(tran_coord_best - tran_coord_worst)
        trans_coord = w*tran_coord_worst + shift_coord
        
        new_rec_extent = rec_extent_worst
        new_lig_extent = lig_extent_worst

        if self.receptor.num_nmodes > 0:
            delta_x = c1*rng.random()*(rec_extent_gbest - rec_extent_worst) + c2*rng.random()*(rec_extent_best - rec_extent_worst)
            n = np.linalg.norm(delta_x)
            if not np.allclose([0.0], [n]):
                delta_x *= (self.step_nmodes / n)
                new_rec_extent += delta_x
                new_rec_extent = np.clip(new_rec_extent, anm.MIN_EXTENT, anm.MAX_EXTENT)
        
        if self.ligand.num_nmodes > 0:
            delta_x = (lig_extent_best - lig_extent_worst) * rng.random()
            delta_x = c1*rng.random()*(lig_extent_gbest - lig_extent_worst)  + c2*rng.random()*(lig_extent_best - lig_extent_worst)
            n = np.linalg.norm(delta_x)

            if not np.allclose([0.0], [n]):
                delta_x *= (self.step_nmodes / n)
                new_lig_extent += delta_x
                new_lig_extent = np.clip(new_lig_extent, anm.MIN_EXTENT, anm.MAX_EXTENT)
        
        return (quart_new, trans_coord, new_rec_extent, new_lig_extent)


    def local_search_one_memeplex_test(self, args):
        """
            im: memeplex_idx
        """
        im, iter_idx = args 
        memeplex = self.memeplexes[im]
        extracted_conformation_data = {int(item):self.conformation_data.get(item) for item in memeplex}
        
        for idx in range(self.no_of_mutation):
            logger.info(f"Iteration {iter_idx} -- Local search of Memeplex {im + 1}: Mutation {idx + 1}/{self.no_of_mutation}")
            unique_id = self.frogs + im + 1
            rValue = rng.random(self.FrogsEach) * self.weights                      # random value with probability weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])                  # index of selected frogs in memeplex
            submemeplex = memeplex[subindex] 

            #--- Improve the worst frog's position ---#
            # Learn from local best Pb #
            Pb = extracted_conformation_data[int(submemeplex[0])]                               # mark the best frog in submemeplex
            Pw = extracted_conformation_data[int(submemeplex[self.q - 1])]                      # mark the worst frog in submemeplex
            
            quart_new, trans_coord, new_rec_extent, new_lig_extent = self.new_step_test(Pb, Pw)
            censorship = False
            
            # Check feasible space and the performance #
            if np.linalg.norm(self.receptor.COM - trans_coord) <= self.omega:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): Learn from local best Pb")
                args = self.transform_complex(unique_id, quart_new, trans_coord, new_rec_extent, new_lig_extent)
                results = self.find_score(args)

                if results[0] < Pw[0]:
                    censorship = True
            else: 
                censorship = True

            if censorship:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): score didn't improve... generating a new frog")
                params = self.generate_one_frog(unique_id)
                results = self.find_score(params)

            shutil.move(os.path.join('poses/', 'out'+str(int(unique_id))+'.pdb'), os.path.join('poses/', 'out'+ str(int(submemeplex[self.q-1])) + '.pdb'))
            extracted_conformation_data[int(submemeplex[self.q-1])] = [results[0], results[2], results[3], results[4], results[5]]
            memeplex = np.array(sorted(extracted_conformation_data, key = lambda x: extracted_conformation_data[x][0], reverse=True))
            logger.info(f"Iteration {iter_idx} -- Local search of Memeplex {im + 1}: pose moved to out{int(submemeplex[self.q-1])}.pdb ::: Mutation {idx + 1}/{self.no_of_mutation} finished")

        return (extracted_conformation_data, im, memeplex)

    def local_search_one_memeplex(self, args):
        """
            im: memeplex_idx
        """
        im, iter_idx = args 
        memeplex = self.memeplexes[im]
        extracted_conformation_data = {int(item):self.conformation_data.get(item) for item in memeplex}
        
        for idx in range(self.no_of_mutation):
            logger.info(f"Iteration {iter_idx} -- Local search of Memeplex {im + 1}: Mutation {idx + 1}/{self.no_of_mutation}")
            unique_id = self.frogs + im + 1
            rValue = rng.random(self.FrogsEach) * self.weights                      # random value with probability weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])                  # index of selected frogs in memeplex
            submemeplex = memeplex[subindex] 

            #--- Improve the worst frog's position ---#
            # Learn from local best Pb #
            Pb = extracted_conformation_data[int(submemeplex[0])]                               # mark the best frog in submemeplex
            Pw = extracted_conformation_data[int(submemeplex[self.q - 1])]                      # mark the worst frog in submemeplex
            
            quart_new, trans_coord, new_rec_extent, new_lig_extent = self.new_step(
                Pb[1], Pb[2], Pb[3], Pb[4], Pw[1], Pw[2], Pw[3], Pw[4])

            globStep = False
            censorship = False
            
            # Check feasible space and the performance #
            if np.linalg.norm(self.receptor.COM - trans_coord) <= self.omega:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): Learn from local best Pb")
                args = self.transform_complex(unique_id, quart_new, trans_coord, new_rec_extent, new_lig_extent)
                results = self.find_score(args)

                if results[0] < Pw[0]:
                    censorship = True
                    # globStep = True
            else: 
                censorship = True
                # globStep = True

            if globStep:
                logger.info(
                    f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): score didn't improve... Learn from global best Pb")
                quart_new, trans_coord, new_rec_extent, new_lig_extent = self.new_step(
                    self.Frog_gb[1], self.Frog_gb[2], self.Frog_gb[3], self.Frog_gb[4], Pw[1], Pw[2], Pw[3], Pw[4])
                
                if np.linalg.norm(self.receptor.COM - trans_coord) <= self.omega:
                    args = self.transform_complex(unique_id, quart_new, trans_coord, new_rec_extent, new_lig_extent)
                    results = self.find_score(args)
                    if results[0] < Pw[0]:
                        censorship = True
                else:
                    censorship = True

            if censorship:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}(out{unique_id}.pdb): score didn't improve... generating a new frog")
                params = self.generate_one_frog(unique_id)
                # quart_new = self.new_quaternion_change(Pw[1])
                # args = self.transform_complex(unique_id, quart_new, Pw[2], Pw[3], Pw[4])
                results = self.find_score(params)

            shutil.move(os.path.join('poses/', 'out'+str(int(unique_id))+'.pdb'), os.path.join('poses/', 'out'+ str(int(submemeplex[self.q-1])) + '.pdb'))
            extracted_conformation_data[int(submemeplex[self.q-1])] = [results[0], results[2], results[3], results[4], results[5]]
            memeplex = np.array(sorted(extracted_conformation_data, key = lambda x: extracted_conformation_data[x][0], reverse=True))
            logger.info(f"Iteration {iter_idx} -- Local search of Memeplex {im + 1}: pose moved to out{int(submemeplex[self.q-1])}.pdb ::: Mutation {idx + 1}/{self.no_of_mutation} finished")

        return (extracted_conformation_data, im, memeplex)

    def local_search(self, iter_idx):
        self.Frog_gb = self.conformation_data[int(self.memeplexes[0][0])]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            doargs = [[im, iter_idx] for im, _ in enumerate(self.memeplexes)]
            results = executor.map(self.local_search_one_memeplex, doargs)
            
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
        self.receptor = Complex(rec_name, data_path, anm.DEFAULT_NMODES_REC)
        self.ligand = Complex(lig_name, data_path, anm.DEFAULT_NMODES_LIG)
        self.ligand.move_to_origin(inplace=True)

        self.receptor.n_modes = anm.calculate_nmodes(self.receptor.pdb_file, anm.DEFAULT_NMODES_REC)
        self.ligand.n_modes = anm.calculate_nmodes(self.ligand.pdb_file, anm.DEFAULT_NMODES_LIG)
        
        self.receptor.dfire_objects = dfire.get_dfire_objects(self.receptor.structure)
        self.ligand.dfire_objects = dfire.get_dfire_objects(self.ligand.structure)
        
        self.dfire_model = dfire.DFIRE()

        self.generate_init_population()
        self.memeplexes = self.sort_frog()
        
        self.omega = 1.25 * self.receptor_max_diameter
        self.step_size = 1

        for idx in range(self.no_of_iteration):
            logger.info(f"Local Search: {idx+1}/{self.no_of_iteration}")
            self.local_search(idx+1)
            self.shuffle_memeplexes()
            # if rng.random() < 0.65:	
            #     self.shuffle_memeplexes()

        directory = "native_" + protein_name
        final_path = os.path.join("./", directory)
        logger.info(f"Creating a new directory - {final_path}")
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        
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
        self.receptor = Complex(rec_name, data_path, anm.DEFAULT_NMODES_REC)
        self.ligand = Complex(lig_name, data_path, anm.DEFAULT_NMODES_LIG)
        self.ligand.move_to_origin(inplace=True)

        self.receptor.n_modes = anm.calculate_nmodes(self.receptor.pdb_file, anm.DEFAULT_NMODES_REC)
        self.ligand.n_modes = anm.calculate_nmodes(self.ligand.pdb_file, anm.DEFAULT_NMODES_LIG)
        
        self.receptor.dfire_objects = dfire.get_dfire_objects(self.receptor.structure)
        self.ligand.dfire_objects = dfire.get_dfire_objects(self.ligand.structure)
        
        self.dfire_model = dfire.DFIRE()

        self.generate_init_population()
        self.memeplexes = self.sort_frog()
        
        self.omega = 1.5 * self.receptor_max_diameter
        self.step_size = 1

    def run_remaining(self):
        for idx in range(5):
            logger.info(f"Local Search: {idx+1}/{5}")
            self.local_search(idx)
            # if rng.random() < 0.65:	
            #     self.shuffle_memeplexes()
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

    sfla = SFLA(frogs=480, mplx_no=40, no_of_iteration=n, no_of_mutation=12, q=8)  # TODO: 400, 40, n, 10, 6 
    #sfla = SFLA(frogs=50, mplx_no=10, no_of_iteration=n, no_of_mutation=2, q=4)
    sfla.run_sfla(str(args.pdb), protein_name, rec_lig_name[0], rec_lig_name[1])