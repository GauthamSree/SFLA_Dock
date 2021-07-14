import os
import numpy as np
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import min_dist

from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
warnings.simplefilter('ignore', PDBConstructionWarning)


class Complex:

    def __init__(self, name, path, num_nmodes):
        self.path = path
        self.chain = [i for i in name]
        self.name = name + '_model_st'
        self.parser = PDBParser()
        self.recognized_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                           'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'NH', 'OC']
        self.atom_types = [['N'], ['CA'], ['C'], ['O'], ['GLYCA'],
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
        self.num_nmodes = num_nmodes
        self.n_modes: np.ndarray = np.array([])
        self.run()

    def pdb_preprocess(self, file):
        self.structure = self.parser.get_structure('1bth', os.path.join(self.path, file + ".pdb"))
        atom_coord = np.empty((0,3))
        atom_mass = []

        for model in self.structure:
            for chain in model:
                if chain.id in self.chain:
                    for residue in chain:
                        for atom in residue:
                            if  not atom.name == 'H':
                                atom_coord = np.append(atom_coord,[atom.get_coord()], axis=0)
                                atom_mass.append(atom.mass)

        self.atom_coord = atom_coord
        self.atom_mass = np.array(atom_mass)
        self.COM = self.find_center_of_mass()

        idx = 0
        with open(os.path.join(self.path, file + ".pdb"), "r") as pdb_in: 
                    with open(file + "1.pdb", "w") as out: 
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
                                l[6] = ("%8.3f" % (float(self.atom_coord[idx][0]))).rjust(8)
                                l[7] = ("%8.3f" % (float(self.atom_coord[idx][1]))).rjust(8)
                                l[8] = ("%8.3f" % (float(self.atom_coord[idx][2]))).rjust(8)
                                l[9] = ("%6.2f" % (float(line[55:60]))).rjust(6)
                                l[10] = ("%6.2f" % (float(line[60:66]))).ljust(6)
                                out.write(
                                    "{0} {1}  {2} {3} {4}{5}    {6}{7}{8}{9}{10}".format(
                                        l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10]
                                    )
                                )
                                out.write("\n")
                                atmno += 1
                                idx += 1
        return file + "1.pdb"


    def chaindef(self):
        self.structure = self.parser.get_structure('1bth', self.pdb_file)
        residue_id = []
        boundary_residue_coord = np.empty((0,3))
        atom_coord = np.empty((0,3))
        boundary_residue_id = []
        boundary_residue_name = []
        
        for model in self.structure:
            surface = get_surface(model)
            for chain in model:
                if chain.id in self.chain:
                    for residue in chain:
                        cx, cy, cz = 0.0, 0.0, 0.0
                        count = 0
                        atom_set = np.empty((0,3))
                        for atom in residue:
                            if  not atom.name == 'H':
                                ax, ay, az = atom.get_coord()
                                atom_set = np.append(atom_set, [atom.get_coord()], axis=0)
                                atom_coord = np.append(atom_coord, [atom.get_coord()], axis=0)
                                cur_atom = residue.get_resname() + atom.name
                                for typ in self.atom_types:
                                    if  cur_atom in typ or atom.name in ['N','CA','C','O']:	 #typ:#atom.name now added
                                        cx += ax
                                        cy += ay
                                        cz += az
                                        count += 1
                                    else:
                                        pass
                        cx /= float(count)
                        cy /= float(count)
                        cz /= float(count)
                        
                        residue_id.append(str(residue.get_id()[1]) + residue.get_id()[2])

                        within_range = False       # Check whether any of the atoms in the residue are at a distance 2 A or less from surface

                        for atm in atom_set:
                            if min_dist(atm, surface) < 2:
                                within_range = True
                                break                        
                       
                        if within_range:
                            boundary_residue_coord = np.append(boundary_residue_coord, [[cx, cy, cz]], axis=0)
                            boundary_residue_id.append(str(residue.get_id()[1]) + residue.get_id()[2])
                            boundary_residue_name.append(residue.get_resname())
        
        self.coord = boundary_residue_coord
        self.residue_name = boundary_residue_name
        self.residue_id = boundary_residue_id
        self.atom_coord = atom_coord


    def find_center_of_mass(self, atom_coord=None):
        c = (atom_coord.T * self.atom_mass).T if atom_coord is not None else (self.atom_coord.T * self.atom_mass).T
        total_mass = np.sum(self.atom_mass)
        com = np.divide(np.sum(c, axis=0), total_mass)
        return com

    def run(self):
        self.pdb_file = self.pdb_preprocess(file=self.name)
        self.chaindef()

    def move_to_origin(self, inplace=False, atom_coord:np.ndarray=None):
        """Moves the structure to the origin of coordinates"""
        com = self.find_center_of_mass(atom_coord) if atom_coord is not None else self.find_center_of_mass()
        
        if np.allclose(com, 1e-14):
            if not inplace and atom_coord is None:
                return self.atom_coord.copy()
            if not inplace and atom_coord is not None:
                return atom_coord, com
        
        if inplace and atom_coord is None:
            self.atom_coord = self.translation(self.atom_coord, -com)
            self.COM = self.find_center_of_mass() 

        else:
            if atom_coord is not None:
                return self.translation(atom_coord, -com), com
            else:
                return self.translation(self.atom_coord, -com)

    def move_back(self, atoms, center):
        if np.allclose(center, 1e-14):
            return atoms
        return self.translation(atoms, center)

    def use_normal_modes(self, anm_extent):
        pose = self.atom_coord.copy()
        if self.num_nmodes > 0:
            for i in range(self.num_nmodes):
                pose += self.n_modes[i] * anm_extent[i]
        return pose
    
    def translation(self, atoms, trans_coord):
        return atoms + trans_coord

    def rotation(self, q, atom_coord:np.ndarray=None):
        if atom_coord is None:
            atm_org = self.move_to_origin()
            atm_rot = np.array([q.rotate(i) for i in atm_org])
            final = self.move_back(atm_rot, self.COM)
        else:
            atm_org, com = self.move_to_origin(atom_coord=atom_coord)
            atm_rot = np.array([q.rotate(i) for i in atm_org])
            final = self.move_back(atm_rot, com)
        
        return final

    def tranformation(self, q=None, trans_coord=None, do_rot_trans=True, do_anm=False, anm_extent:np.ndarray=None):
        if do_anm:
            anm_coordinates = self.use_normal_modes(anm_extent) 
            if do_rot_trans:
                new_coord = self.rotation(q=q, atom_coord=anm_coordinates)
                coordinates = self.translation(new_coord, trans_coord)
            else:
                coordinates = anm_coordinates
        if not do_anm and do_rot_trans:
            new_coord = self.rotation(q)
            coordinates = self.translation(new_coord, trans_coord)
        
        if not do_anm and not do_rot_trans:
            coordinates = self.atom_coord
        
        return coordinates

    def __repr__(self):
        return f"Complex: name = {self.name}, pdb_file_name = {self.pdb_file}"

    def __str__(self):
        return f"Complex: name = {self.name}, pdb_file_name = {self.pdb_file}"