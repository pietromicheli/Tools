#Tools
import sys
import os
import subprocess
import json
import numpy                    as np
import urllib.request           as urllib
from xml.dom                    import minidom
from scipy.spatial              import distance
from Bio                        import pairwise2
from Bio.SubsMat.MatrixInfo     import blosum62
from Bio.Blast                  import NCBIWWW
from Bio.PDB.PDBParser          import PDBParser
from Bio.PDB                    import PDBIO
from Bio.PDB.vectors            import Vector
from Bio.PDB.Superimposer       import Superimposer
from modeller                   import *
from modeller.optimizers        import molecular_dynamics, conjugate_gradients
from modeller.automodel         import autosched
from Tools.paths import *

from urllib.request import urlopen

mgltools_path=''
vina_executable_path =''
bp_center_reference_file='/autogrid_reference_files/'
dlscore_path=''


class protein_recognition:
    '''tools to retrive protein information'''
    def get_seq(structure_filename, chain):
        '''pdb file as input + chain
        give us the sequence as string (converts 3 leter code to one letter code)'''
        #return the one-letter-code sequence of a structure
        #NB: specified chain must be in the FIRST model of the structure

        letters = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H',
                   'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
                   'TYR':'Y','VAL':'V'}
        seq = ""

        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure('query', structure_filename)

        chain = structure[0][chain]
        for residue in chain:
            if residue.id[0] == ' ':
                seq = seq+letters[residue.get_resname()]

        return seq

    def get_uniprot_entry(structure_filename, chain):
        '''pdb file as input + chain
        gives the uniport id corresponding the the protein of the file'''
        seq = protein_recognition.get_seq(structure_filename, chain)
        results = NCBIWWW.qblast("blastp", "swissprot", seq, format_type="Text",hitlist_size=5 )

        '''este codigo pode ser melhorado'''
        #uniprot_entry = results[34].split(sep='.')[0]

        with open("my_blast.txt", "w") as out:
            out.write(results.read())

        results.close()

        with open("my_blast.txt", 'r') as out:
            for i,line in enumerate(out):
                if i == 34:
                    refseq_entry = line.split(sep='.')[0]
                    out.close()
                    break

        os.remove("my_blast.txt")
        return refseq_entry

    def retrive_geneid(uniprot_entry):
        '''uniprot id as input and give us the gene id/name of the protein'''
        uniprot_url_query = "https://www.ebi.ac.uk/proteins/api/proteins/"+uniprot_entry
        html_data = urllib.urlopen(uniprot_url_query)
        datas = json.loads(html_data.read())

        organism = datas["organism"]["names"][0]["value"]
        if organism != "Homo sapiens":
            print("\nWARNING:   This is not a Human gene!\n")

        gene_id = datas["gene"][0]["name"]["value"]

        return gene_id

    def retrive_uniprot_seq(uniprot_entry):
        '''uniport id as input and give us the original uniprot sequence of protein (download FASTA but return as string)'''
        uniprot_url = "https://www.uniprot.org/uniprot/"
        query_url = uniprot_entry+".fasta"
        html_data = urllib.urlopen(uniprot_url+query_url)

        seq_fasta = html_data.read().decode("utf-8")

        #transform fasta sequence into straight sequence string
        fasta_splitted = seq_fasta.split(sep='\n')
        seq_str = ''.join(fasta_splitted[1:])

        return seq_str

class find_compounds:

    def retrive_approved_drugs(uniprot_entry, ligands_directory):

        '''uniprot id and an output directory. Calls indirectly drugbank and give us the sdf structures of all approved drugs towards the protein. '''
        mychem_url_query = "http://mychem.info/v1/query?q=drugbank.targets.uniprot:"+uniprot_entry+"%20AND%20drugbank.groups:approved&fields=drugbank.id,drugbank.name"
        html_data = urlopen(mychem_url_query)
        datas = json.loads(html_data.read())

        #retrive SDF structures from DrugBank
        for hit in datas["hits"]:

            DrugBank_accession = hit["drugbank"]["id"]
            ligand_name = hit["drugbank"]["name"]
            drugbank_ligand_structure_url = "https://www.drugbank.ca/structures/small_molecule_drugs/"+DrugBank_accession+".sdf?type=3d"
            print(drugbank_ligand_structure_url)
            html_data = urllib.urlopen(drugbank_ligand_structure_url)

            #check if there are spaces in the name of the ligand
            if len(ligand_name.split()) > 1:
                ligand_name = ''.join(ligand_name.split())

            with open(ligands_directory+'/'+ligand_name+'.sdf', "w") as f:
                f.write(html_data.read().decode("utf-8"))

        #if no drug-bank approved drugs were founded
        if not datas["hits"]:

            print("\nWARNING: NO APPROVED DRUGS FOUNDED\n")


class find_variants:
    '''find the variants of the strucutre'''
    def retrive_exac_variants(gene_id):
        '''gene id as input. we can call the function retrive_geneid to get the gene id. Give us a list of all of annotated missense variant
        for the gene.    p.Arg300Lys'''
        #retrive all the missense variations annotated in the ExAc browser for the passed gene

        exac_url = "http://exac.hms.harvard.edu/"
        api_query_url = "/rest/awesome?query="+gene_id+"&service=variants_in_gene"

        html_data_query = urllib.urlopen(exac_url+api_query_url)

        variants = json.loads(html_data_query.read())
        variants_list = []

        for variant in variants:
            if variant["category"] == "missense_variant" and variant["major_consequence"] == "missense_variant":
                variants_list.append(variant["HGVSp"])


        return variants_list

    def LBD_variants_filter(receptor_filename, chain, family, variants_list):
        '''pdb file as input + chain of our subunit + family + python list of HGVSp variants (from previous function).
        give us a python list of all the input variants che stano nel biding site.

        family = GPCR or IGR (ionotropic glutamatergic receptor)

        chain in capital'''
        #given a receptor leading to the specified family and a list of variants (in the HGVSp notation)
        #annotated for the specified chain, return only the variants wich lay on the Ligand Binding Domain of the passed protein's structure.

        #N.B. : a preventive renumeration of the structure is recommended

        def evaluate_distance(center_coordinates, atoms_coordinates, max_distance):

            #if the distance from the passed center of (at least) one of the passed atoms is lower than max_distance return true, false otherwise
            for atom_coordinates in atoms_coordinates:

                d = distance.euclidean(atom_coordinates, center_coordinates)
                if d < max_distance:
                    return True

            return False

        #find the center of the Binding Pocket with bp_center
        center_coordinates = bp_center.bp_center(receptor_filename, chain, family)
        #set maximum allowed distance (Amstrong)
        max_distance = 22

        #parse the structure
        parser = PDBParser(PERMISSIVE=1)
        structure =  parser.get_structure("", receptor_filename)

        #NB: passed chain must be in the first model of the structure
        chain = structure[0][chain]
        structure_res_numbers = []
        filtered_variants_list = []

        for residue in chain:
            structure_res_numbers.append(list(residue.id)[1])


        for variant in variants_list:
            variant_res_number = int(variant[5:len(variant)-3])

            if variant_res_number in structure_res_numbers:
                atoms_coordinates = []

                for atom in chain[variant_res_number]:
                    atoms_coordinates.append(atom.get_coord())

                if evaluate_distance(center_coordinates, atoms_coordinates, max_distance):
                    filtered_variants_list.append(variant)

        return filtered_variants_list


class mutagenesis:

    def mutate(structure_pdb, chain, variant, out_pdb):
        '''pdb file as input + chain + variant in HGVSp notation + output name (inlcuding pdb extension)'''
        #perform the computation af a mutatant model for the passed structure
        #the mutation will be performed over the specified chain, according to the passed HGVSp variant
        #NB : variant must be passed in the HGVSp notation
        '''modeller script'''
        def optimize(atmsel, sched):
            #conjugate gradient
            for step in sched:
                step.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
            #md
            refine(atmsel)
            cg = conjugate_gradients()
            cg.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)

        #molecular dynamics
        def refine(atmsel):
            # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
            md = molecular_dynamics(cap_atom_shift=0.39, md_time_step=4.0,
                                    md_return='FINAL')
            init_vel = True
            for (its, equil, temps) in ((200, 20, (150.0, 250.0, 400.0, 700.0, 1000.0)),
                                        (200, 600,
                                         (1000.0, 800.0, 600.0, 500.0, 400.0, 300.0))):
                for temp in temps:
                    md.optimize(atmsel, init_velocities=init_vel, temperature=temp,
                                 max_iterations=its, equilibrate=equil)
                    init_vel = False

        #use homologs and dihedral library for dihedral angle restraints
        def make_restraints(mdl1, aln):
           rsr = mdl1.restraints
           rsr.clear()
           s = selection(mdl1)
           for typ in ('stereo', 'phi-psi_binormal'):
               rsr.make(s, restraint_type=typ, aln=aln, spline_on_site=True)
           for typ in ('omega', 'chi1', 'chi2', 'chi3', 'chi4'):
               rsr.make(s, restraint_type=typ+'_dihedral', spline_range=4.0,
                        spline_dx=0.3, spline_min_points = 5, aln=aln,
                        spline_on_site=True)

        residues = {'Ala':'ALA','Arg':'ARG','Asn':'ASN','Asp':'ASP','Cys':'CYS','Glu':'GLU','Gln':'GLN','Gly':'GLY','His':'HIS',
                   'Ile':'ILE','Leu':'LEU','Lys':'LYS','Met':'MET','Phe':'PHE','Pro':'PRO','Ser':'SER','Thr':'THR','Trp':'TRP',
                   'Tyr':'TYR','Val':'VAL'}

        # Set a different value for rand_seed to get a different final model
        env = environ(rand_seed=-49837)

        env.io.hetatm = True
        #soft sphere potential
        env.edat.dynamic_sphere=False
        #lennard-jones potential (more accurate)
        env.edat.dynamic_lennard=True
        env.edat.contact_shell = 4.0
        env.edat.update_dynamic = 0.39

        # Read customized topology file with phosphoserines (or standard one)
        env.libs.topology.read(file='$(LIB)/top_heav.lib')

        # Read customized CHARMM parameter library with phosphoserines (or standard one)
        env.libs.parameters.read(file='$(LIB)/par.lib')

        # Read the original PDB file and copy its sequence to the alignment array:
        mdl1 = model(env, file=structure_pdb)
        ali = alignment(env)
        ali.append_model(mdl1, atom_files=structure_pdb, align_codes=structure_pdb)

        '''pietro start'''
        #extract the position and the substitute residue from the passed HGVSp-notated variant
        respos = variant[5:len(variant)-3]
        restyp = residues[variant[len(variant)-3:]]
        '''pietro stop'''

        #set up the mutate residue selection segment
        s = selection(mdl1.chains[chain].residues[respos])

        #perform the mutate residue operation
        s.mutate(residue_type=restyp)
        #get two copies of the sequence.  A modeller trick to get things set up
        ali.append_model(mdl1, align_codes=structure_pdb)

        # Generate molecular topology for mutant
        mdl1.clear_topology()
        mdl1.generate_topology(ali[-1])

        # Transfer all the coordinates you can from the template native structure
        # to the mutant (this works even if the order of atoms in the native PDB
        # file is not standard):
        #here we are generating the model by reading the template coordinates
        mdl1.transfer_xyz(ali)

        # Build the remaining unknown coordinates
        mdl1.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')

        #yes model2 is the same file as model1.  It's a modeller trick.
        mdl2 = model(env, file=structure_pdb)

        #required to do a transfer_res_numb
        #ali.append_model(mdl2, atom_files=structure_pdb, align_codes=structure_pdb)
        #transfers from "model 2" to "model 1"
        mdl1.res_num_from(mdl2,ali)

        #It is usually necessary to write the mutated sequence out and read it in
        #before proceeding, because not all sequence related information about MODEL
        #is changed by this command (e.g., internal coordinates, charges, and atom
        #types and radii are not updated).

        mdl1.write(file='temp.tmp')
        mdl1.read(file='temp.tmp')

        #set up restraints before computing energy
        #we do this a second time because the model has been written out and read in,
        #clearing the previously set restraints
        make_restraints(mdl1, ali)

        #a non-bonded pair has to have at least as many selected atoms
        mdl1.env.edat.nonbonded_sel_atoms=1

        sched = autosched.loop.make_for_model(mdl1)

        #only optimize the selected residue (in first pass, just atoms in selected
        #residue, in second pass, include nonbonded neighboring atoms)
        #set up the mutate residue selection segment
        s = selection(mdl1.chains[chain].residues[respos])

        mdl1.restraints.unpick_all()
        mdl1.restraints.pick(s)

        s.energy()

        s.randomize_xyz(deviation=4.0)

        mdl1.env.edat.nonbonded_sel_atoms=2
        optimize(s, sched)

        #feels environment (energy computed on pairs that have at least one member
        #in the selected)
        mdl1.env.edat.nonbonded_sel_atoms=1
        optimize(s, sched)

        s.energy()

        #give a proper name
        mdl1.write(file=out_pdb)

        #delete the temporary file
        os.remove('temp.tmp')

        return out_pdb


class bp_center:
    '''trova il centro del ligand binding domain'''
    #reference_files = bp_center_reference_files

    def get_seq(res_list):

        #return the one-letter-code sequence of a list of residues
        letters = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H',
                   'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
                   'TYR':'Y','VAL':'V'}
        residues = []
        seq = ""

        for res in res_list:
            if res.id[0] == ' ':
                seq = seq + letters[res.get_resname()]

        return seq

    def superimpose_get_rotranrms(fixed, moving):

        # 'fixed' and 'moving' are lists of Atom objects.
        # The moving atoms will be put on the fixed atoms.
        # return the transformation matrices

        sup = Superimposer()
        sup.set_atoms(fixed, moving)

        # calculate rot and tran matrices.

        rot,tran = sup.rotran
        rms = sup.rms

        return rms, rot, tran

    def allign_3D(ref_list_atoms, ref_list_res, targ_list_res):

        seq1 = bp_center.get_seq(targ_list_res)
        seq2 = bp_center.get_seq(ref_list_res)

        alignment = pairwise2.align.globalds(seq1, seq2, blosum62, -10, -.5)[0]
        score = alignment[2]

        #set the two atoms lists to use to find the transformation matrix
        ref_list = []
        targ_list = []

        #create a list of indices iterating over alignment, where each element contains a couple of value
        #(start and stop of the continuous match in the alignment)

        start_tar = 0
        stop_tar = 0
        start_ref = 0
        stop_ref = 0
        indexes_tar =[]
        indexes_ref = []
        continuous = False
        count_tar = 0
        count_ref = 0

        max = 0

        for i in range(len(alignment[1])):

            if (alignment[0])[i] != '-' :
                count_tar += 1

            if (alignment[1])[i] != '-' :
                count_ref += 1

            if (alignment[0])[i] != '-' and (alignment[1])[i] != '-' :
                if continuous == False:
                    start_tar = count_tar
                    start_ref = count_ref
                    continuous = True
            else:
                if continuous == True:
                    stop_tar = count_tar
                    indexes_tar.append([start_tar,stop_tar])
                    stop_ref = count_ref
                    indexes_ref.append([start_ref,stop_ref])
                    continuous = False
                    if (stop_tar - start_tar) > max :
                        max = stop_tar-start_tar

        #check if the alignment produced a perfect match. if true perform the superimposition on the whole atom lists,
        #otherwise perform the superimposition on the atoms leading to continuous-matching subsequences of the alignment:
        #set k as the minimum length of the alignment's continuous subsequences you want to consider.
        #NB! max is used to fix the maximum value that k can assume, and refears to the maximum continuous subsequence length.

        if len(indexes_tar) == 0 and count_ref == len(ref_list_res):

            for res in targ_list_res:
                for atom in res:
                    targ_list.append(atom)

            ref_list = ref_list_atoms

        else:
            k = max -10

            #extract from the target structure the atoms leading to continuous match
            #subsequences of the alignment with length >= keys

            for index_tar, index_ref in zip(indexes_tar, indexes_ref):

                if index_tar[1]-index_tar[0] >= k :
                    for i in range(index_tar[0],index_tar[1]):

                        for atom in targ_list_res[i]:
                            targ_list.append(atom)

                    for i in range(index_ref[0],index_ref[1]):

                        for atom in ref_list_res[i]:
                            ref_list.append(atom)

        #resize the two lists to perform superimposition

        if len(targ_list)>=len(ref_list):
            targ_list = targ_list[:len(ref_list)]

        else:

            ref_list = ref_list[:len(targ_list)]

        #try superimpose
        rms,rot,tran = bp_center.superimpose_get_rotranrms(targ_list,ref_list)

        return rms, rot, tran, score

    def bp_center(receptor_pdb, chain, family, verbose=False):

        #perform an automatic detection of the orthosteric binding pocket's center for the passed chain of the passed structure

        #set the reference files directory
        #reference_files = bp_center.reference_files
        reference_files = '/home/rribeiro/Projects/Tools/autogrid_reference_files'


        #NEW RECEPTORS FAMILIES SHOULD BE ADDED HERE!

        families = {
                        'IGR' : [reference_files+'/IGr_Ref.pdb', reference_files+'/IGr_Ref_grids.txt'],
                        'GPCR' : [reference_files+'/GPCr_Ref.pdb', reference_files+'/GPCr_Ref_grids.txt']
        }


        if family not in families:
            print("\nArgoument Error: -family argoument should be ", families.keys(),'\n\n')

        else:
            reference_structures_filename = (families[family])[0]
            reference_grids_filename = (families[family])[1]

        parser = PDBParser(PERMISSIVE=1)

        s1 = parser.get_structure("tar", receptor_pdb)
        s2 = parser.get_structure("ref", reference_structures_filename)

        tar_chain = s1[0][chain]
        tar_atoms_list = []
        tar_res_list = []

        for residue in tar_chain:
            tar_res_list.append(residue)
            for atom in residue:
                tar_atoms_list.append(atom)

        ref_atoms_dict = {}
        ref_res_dict = {}

        for ref_chain in s2[0]:
            residues = []
            atoms = []
            ref_res_dict.update({ref_chain.id : residues})
            ref_atoms_dict.update({ref_chain.id : atoms})
            for residue in ref_chain:
                ref_res_dict[ref_chain.id].append(residue)
                for atom in residue:
                    ref_atoms_dict[ref_chain.id].append(atom)

        grids_file_ref = open(reference_grids_filename, "r")

        #create a dictionary containing the coordinates of the reference grids (as vectors) which can be
        #modified during the execution in order to not to modify the original reference grids file,
        #and a dictionary that will contain the transformed coordinates of the grids.

        ref_grids_dic = {}
        for line in grids_file_ref :

            toks = line.split()
            chain_id = toks[0]
            coor = Vector(toks[2],toks[3],toks[4])
            ref_grids_dic.update({chain_id : coor})

        #Start the detection

        #set a reasonable high value to minimize the rmsd
        score_opt = 0
        rms_opt = 1000

        for ref_chain in s2[0]:

            #create copies in order to preserve the integrity of the original chains
            ref_atoms_list = ref_atoms_dict[ref_chain.id]
            ref_res_list = ref_res_dict[ref_chain.id]

            rms,rot,tran,score = bp_center.allign_3D(ref_atoms_list,ref_res_list,tar_res_list)

            if rms < rms_opt:
                rms_opt = rms
                rot_opt, tran_opt = rot, tran
                score_opt = score
                opt = ref_chain.id

        #set a threshold for the alignment score to judge the goodness of the calculated alignment
        if score_opt < 80 and rms_opt > 7:

            print('Error: no good structural alignment found for chain ', chain)
            print(score_opt,rms_opt)
        else:

            #read the reference coordinates file and transform the relative coordinates currently stored in the dictionary,
            #then write them to the output target's grids coordinates file.
            #NB: this is important , because all the transformations performed of the reference chains change the
            #position of the chains themselves, but not the relative grids coordinates!

            ref_grid_coor = ref_grids_dic[opt]
            targ_grid_coor = list(Vector.right_multiply(ref_grid_coor, rot_opt) + tran_opt)

            if verbose :

                #print summary
                print("###############################################################")
                print("                                                               ")
                print("Target chain '"+chain+"' Summary :                             ")
                print("                                                               ")
                print("reference chain used : '"+opt+"'"                               )
                print("calculated rmsd :",rms_opt,                                     )
                print("calculated score alignment :",score_opt,                        )
                print("grid center coordinates : ",targ_grid_coor,                     )
                print("                                                               ")
                print("###############################################################")
                print("                                                               ")

            #return the coordinates of the binding pocket's center for the passed chain
            return targ_grid_coor


class automatic_docking:

    mgltools = mgltools_path
    vina_exe = vina_executable_path
    dlscore = dlscore_path

    def automatic_docking(prepared_receptor_pdbqt, prepared_ligand_pdbqt, bp_center_coordinates, output_location_pdbqt):

        #perform an automatic docking of the passed prepared receptor.pdbqt with the passed prepared ligand.pdbqt over the passed chain
        #the grid of Vina will be centered in the coordinates passed by the bp_center_coordinates list [x,y,z]
        #pdbqt output will be saved in the specified output_location.pdbqt


        # set grid center and dimension.
        x = str(bp_center_coordinates[0])
        y = str(bp_center_coordinates[1])
        z = str(bp_center_coordinates[2])
        size_x = '22'
        size_y = '22'
        size_z = '22'

        #run Autodock Vina, using the computed coordinates to set the grids.
        subprocess.call([automatic_docking.vina_exe, "--receptor",prepared_receptor_pdbqt,"--ligand",prepared_ligand_pdbqt,"--center_x",x,"--center_y",y,"--center_z",z,"--size_x",size_x,"--size_y",size_y,"--size_z",size_z,"--out",output_location_pdbqt])

        # return the output pdbqt file
        return output_location_pdbqt

    def prepare_receptor(receptor_filename, output_location_pdbqt):

        receptor_path, receptor_extension = receptor_filename.split(sep='.')

        if receptor_extension == 'sdf':

            os.system('babel '+receptor_filename+' '+receptor_path+'.pdb')

        elif receptor_extension == 'pdb' or receptor_extension == 'pdbqt': pass

        else:
            print('Error: file extension should be pdb, sdf or pdbqt')


        #Convert pdb to pdbqt with MGLTOOLS

        if receptor_extension != 'pdbqt':

            subprocess.call([automatic_docking.mgltools+"\\python",automatic_docking.mgltools+"\\Lib\\site-packages\\AutoDockTools\\Utilities24\\prepare_receptor4.py", "-r",(receptor_path+".pdb"),"-o",(output_location_pdbqt),'-A','hydrogens'])

        #return the converted receptor.pdbqt
        return output_location_pdbqt

    def prepare_ligand(ligand_filename, output_location_pdbqt):

        ligand_path, ligand_extension = ligand_filename.split(sep='.')

        if ligand_extension == 'sdf':

            os.system('babel '+ligand_filename+' '+ligand_path+'.pdb')

        elif ligand_extension == 'pdb' or ligand_extension == 'pdbqt': pass

        else:
            print('Error: file extension should be pdb, sdf or pdbqt')


        #Convert pdb to pdbqt with MGLTOOLS

        if ligand_extension != 'pdbqt':

            subprocess.call([automatic_docking.mgltools+"\\python",automatic_docking.mgltools+"\\Lib\\site-packages\\AutoDockTools\\Utilities24\\prepare_ligand4.py", "-l",(ligand_path+".pdb"),"-o",(output_location_pdbqt),'-A','hydrogens'])

        #return the converted ligand.pdbqt
        return output_location_pdbqt

    def compute_dlscore(prepared_receptor_pdbqt, vina_results_pdbqt):

        vina_best_result = handle_pdb.vina_best_result(vina_results_pdbqt, 'best_temp.pdbqt')

        #Run DLScore
        subprocess.call(["python",automatic_docking.dlscore,"--receptor",prepared_receptor_pdbqt,"--ligand",vina_best_result,"--vina_executable",automatic_docking.vina_exe,"--network_type", "refined","--verbose","1"])
        os.remove(vina_best_result)

        #read scores from out.csv
        with open("out.csv",'r') as out:
            lines = out.readlines()
            algos = lines[0][:len(lines[0])-1].split(sep=',')
            scores = lines[2][:len(lines[2])-1].split(sep=',')

        out = {}
        for algo, score in zip(algos, scores):
            out.update({algo : score})

        os.remove("out.csv")

        return out


class handle_pdb:

    def clean_pdb(file_pdb, out_pdb):

        with open(file_pdb, "r") as input:
            lines = []
            for line in input:
                columns = line.split()
                if columns[0] in ['ATOM', 'TER']:
                    lines.append(line)

        with open("temp.tmp", "w") as output:
            output.write("MODEL1\n")
            output.writelines(lines)
            output.write("ENDMDL\n")
            output.write("END")

        if file_pdb == out_pdb:
            os.remove(file_pdb)

        os.rename("temp.tmp", out_pdb)

        return out_pdb

    def pdbqt2pdb(file_pdbqt, file_pdb):

        #convert pdbqt to pdb
        subprocess.call('babel -i pdbqt '+file_pdbqt+' -o pdb '+file_pdb)

        return file_pdb

    def append_pdb(file1_pdb, file2_pdb, out_filename):

        #append file1_pdb to file2_pdb
        ref = open(file2_pdb,'r')
        app = open(file1_pdb,'r')
        out = open(out_filename,'w+')

        out.write('MODEL 1\n')

        for line in app:
            columns = line.split()
            if columns[0] in ['ATOM','CONNECT']:
                out.write(line)

        out.write('ENDMDL\n')
        out.write('MODEL 2\n')

        for line in ref:
            columns=line.split()
            if columns[0] in ['ATOM','HETATM','CONNECT']:
                out.write(line)

        out.write('ENDMDL\n')
        out.write('END')

    def merge_pdbqts(pdbqt_filename1, pdbqt_filename2, pdb_out_filename):

        #merge two pdbqt files in a pdb file
        pdbqt_basename1 = (os.path.splitext(os.path.basename(pdbqt_filename1)))[0]
        pdbqt_basename2 = (os.path.splitext(os.path.basename(pdbqt_filename2)))[0]
        converted_filename1 = pdbqt_basename1+'.pdb'
        converted_filename2 = pdbqt_basename2+'.pdb'
        pdbqt2pdb(pdbqt_filename1, converted_filename1)
        pdbqt2pdb(pdbqt_filename2, converted_filename2)
        append_pdb(converted_filename1, converted_filename2, out_filename)

        os.remove(converted_filename1)
        os.remove(converted_filename2)

    def vina_best_result(pdbqt_result_filename, pdbqt_best_result_filename):

        #extract first result from vina pdbqt result file
        with open(pdbqt_result_filename, "r") as file:
            lines = []

            for line in file:
                lines.append(line)
                columns=line.split()
                if columns[0] == 'ENDMDL': break

        with open(pdbqt_best_result_filename, "w") as out:
            out.writelines(lines)

        return pdbqt_best_result_filename

    def renumerate_structure(structure_pdb, chain, uniprot_entry, structure_pdb_out):

        #Renumerate the residues of the passed structure's chain according with the original uniprot sequence
        #Sequences must be passed as string objects
        #NB: specified chain must be in the FIRST model of the structure

        def recursive_renumbering(new_id, new_ids, chain, dict):

            residue = chain[new_id]
            current_id = list(residue.id)
            index = dict[current_id[1]]
            new_id = new_ids[index]
            if int(current_id[1]) > int(new_id) and new_id in chain:
                recursive_renumbering(new_id, new_ids, chain, dict)

            current_id[1] = new_id
            residue.id = tuple(current_id)

        structure_seq = protein_recognition.get_seq(structure_pdb, chain)
        original_seq_fasta_file = protein_recognition.retrive_uniprot_seq(uniprot_entry)

        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure('query', structure_pdb)

        alignment = pairwise2.align.localdd(original_seq_fasta_file, structure_seq, blosum62, -1000, -1000, -10, -0.5)[0]

        new_ids = []
        for i in range(len(alignment[1])):

            if alignment[1][i] != '-':
                new_ids.append(i+1)

        chain = structure[0][chain]

        #create a residue : structure-relative-position dict
        dict = {}
        for index, residue in enumerate(chain):
            dict.update({residue.id[1] : index})

        #start the re-enumeration from the end of the structure
        for index, residue in reversed(list(enumerate(chain))):

            current_id = list(residue.id)
            new_id = new_ids[index]

            #check if we're trying to use an already exixting residue identifier, if so, first re-enumerate that residue
            if int(current_id[1]) > int(new_id) and new_id in chain:
                recursive_renumbering(new_id, new_ids, chain, dict)

            current_id[1] = new_id
            residue.id = tuple(current_id)


        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(structure_pdb_out)

        return structure_pdb_out
