"""Module to calculate normal modes of a given protein.
"""

import numpy as np
from prody import parsePDB, ANM, extendModel, confProDy

confProDy(verbosity='none') # Disable ProDy output


# Normal modes
DEFAULT_NMODES_REC = 10              # Default number of normal modes to consider for receptor
DEFAULT_NMODES_LIG = 10              # Default number of normal modes to consider for ligand
MIN_EXTENT = 0.1
MAX_EXTENT = 5.0             
DEFAULT_EXTENT_MU = 4.0
DEFAULT_EXTENT_SIGMA = 3.0
DEFAULT_NMODES_STEP = 0.5

def calculate_nmodes(pdb_file_name, n_modes, molecule=None):
    """Calculates Normal modes for a given molecule"""
    prody_molecule = parsePDB(pdb_file_name)
    backbone_atoms = prody_molecule.select('name CA')
    molecule_anm = ANM('molecule backbone')
    molecule_anm.buildHessian(backbone_atoms)
    molecule_anm.calcModes(n_modes=n_modes)

    num_atoms_prody = prody_molecule.numAtoms()

    molecule_anm_ext, molecule_all = extendModel(molecule_anm, backbone_atoms, prody_molecule, norm=True)
    modes = []
    calculated_n_modes = (molecule_anm_ext.getEigvecs()).shape[1]
    try:
        for i in range(calculated_n_modes):
            nm = molecule_anm_ext.getEigvecs()[:, i].reshape((num_atoms_prody, 3))
            modes.append(nm)
    except (ValueError, IndexError) as e:
        print(f"Number of atoms and ANM model differ. Please, check there are no missing nucleotides nor residues. Error: {e}")
    if calculated_n_modes < n_modes:
        # Padding
        for i in range(n_modes - calculated_n_modes):
            modes.append(np.zeros((num_atoms_prody, 3)))

    return np.array(modes)

def write_nmodes(n_modes, output_file_name):
    """Writes the previous calculated n_modes to a binary numpy file"""
    np.save(output_file_name, n_modes)

def read_nmodes(file_name):
    """Reads normal modes from a numpy binary file"""
    return np.load(file_name)