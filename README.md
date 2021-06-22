# SFLADock
Code folder contains the source code of SFLADock
 
**Prerequisites:**

Please install the following packages before running the program:

    biopython
    msms
    pandas
    emboss
    python-tk
    matplotlib
    scikit-learn
    pyquaternion
    pdbtools
    scipy

Alternatively, one can use the py3.yml to install all the dependencies.
conda env create -f py3.yml will serve the purpose.

SFLADock uses van der Waals and electrostatic potentials in meetdock scoring function, which can be found in https://github.com/maxibor/meetdock. 

Save the input files in Data folder. Sample inputs are uploaded in this folder. The folder name 4dn4_LH:M indicates that 4DN4 is the target protein containing L, H, and M chains. Receptor PDB file is named as LH_model_st.pdb and contains L and H chains. Ligand PDB file is named as M_model_st.pdb and contains M chain. Final result will appear in the folder 'poses'
**To run:**
./fpdock.sh
 
  
 fpdock.sh includes the code to analyse the result. You can comment the lines from line number 26 to avoid its execution. For analysis, an additional software DockQ is needed to be downloaded from "https://github.com/bjornwallner/DockQ". 
 
  
 In the shell file, target is set as 4DN4. To dock other inputs, change the target in line number 2. Naming conventions must be followed when trying a new target.

**Reference**
1. https://github.com/maxibor/meetdock
