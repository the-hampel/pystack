import requests
import re
import sys
import numpy as np

#_____________________________________________________________________________________________#

# copyright by https://github.com/noemas/pysodistort! 

#function that will submit a parent (_parent_cif) and a child cif (_child_cif) to ISODISTORT. You can specify if you want to choose the 
#possible bases during runtime (_basis_prompt) or if you want to use a predetermined basis (_basis, _basis_number).
#To do so either specify the basis (_basis) or give the number of the basis (_basis_number)
#If the basis is provided with _basis, it needs to have the following syntax:
#"x1 y1 z1 x2 y2 z2 x3 y3 z3 t" i.e. the first 3 numbers correspond to the first basis vector etc.
#The output will be the final html file

def submit_isodistort(_parent_cif, _child_cif, _basis_prompt = True, _basis = "", _basis_number = None):
    #php urls
    upload_php = "https://stokes.byu.edu/iso/isodistortuploadfile.php"
    form_php = "https://stokes.byu.edu/iso/isodistortform.php"
    
    #prepare the POST parameters for the upload of the parent cif
    parameters = {'input':'uploadparentcif'}
    
    #upload the parent cif file
    file = open(_parent_cif,'rb')
    upload = requests.post(upload_php, data = parameters, files={"toProcess":file})
    
    #get the name of the temporary parent file
    temp_file = re.search(r'VALUE=\"/tmp(.*)', upload.text)[0]
    temp_file = temp_file[:-2]
    temp_file = temp_file[7:]
    
    #submit the temporary parent cif to the form
    parameters = {'filename': temp_file, 'input': 'uploadparentcif'}
    form = requests.post(form_php, data = parameters)
    
    #get the names of the necessary hidden values of the parent cif
    hidden_values = re.findall(r'hidden(.*)', form.text)
    num_of_values = len(hidden_values)
    value_names = [""] * num_of_values
    value_values = [""] * num_of_values
    start_index = 0;
    end_index = 0;
    
    #the relevant parameters are located after the uploadsubgroupcif input specification
    for i in range(0, len(hidden_values)):
        value_names[i] = hidden_values[i].split('"')[2]
        value_values[i] = hidden_values[i].split('"')[4]
        if value_values[i] == "uploadsubgroupcif":
            start_index = i
        elif value_values[i] == "changesearch" and start_index != 0 and i > start_index:
            end_index = i-1
    
    #prepare the POST parameters for the upload of the child cif (these are the parameters of the parent cif)
    parameters = {}
    for i in range(start_index, end_index):
        parameters[str(value_names[i])] = str(value_values[i])
    
    #upload parent and child cif (the parent as parameters, the child as file)
    file = open(_child_cif,'rb')
    upload = requests.post(upload_php, data = parameters, files={"toProcess":file})
    
    #get the name of the temporary child file
    temp_file = re.search(r'VALUE=\"/tmp(.*)', upload.text)[0]
    temp_file = temp_file[:-2]
    temp_file = temp_file[7:]
    
    #give the temporary child file name as an additional parameter
    parameters['filename'] = temp_file
    
    #submit the parent and child cif to the form (the parent as parameters, the child as temporary file)
    form = requests.post(form_php, data = parameters)
    
    bases = re.search(r'\"basisselect\"((.|\n)*?)<br>', form.text)[0]
    #get the number of bases
    num_of_bases = int(bases.count('OPTION') / 2 - 1)
       
        
    #get the basis index if prompting 
    if _basis_prompt:
        if not _basis == "":
            sys.exit("You can't prompt for basis (_basis_prompt) if a basis is specified (_basis)")
        if not _basis_number == None:
            sys.exit("You can't prompt for basis (_basis_prompt) if a basis number is specified (_basis_number)")

        #first print all bases
        for i in range(0, num_of_bases):
            _basis = re.findall(r'VALUE=(.*)', bases)[i]
            _basis = _basis.split("\"")[1]
            print("Basis " + str(i+1) + ": " + str(_basis)) 
        #prompt index
        basis_number = int(input("Enter basis number: ")) 
        if num_of_bases < basis_number:
            sys.exit("You have chosen basis number " + str(basis_number) + " but there are only " + str(num_of_bases) + " bases.")
        
        _basis = re.findall(r'VALUE=(.*)', bases)[basis_number - 1]
        _basis = _basis.split("\"")[1]
        print("Chosen basis: " + str(_basis))
    
    #get the basis if basis number is given
    if _basis_number:
        if _basis_prompt:
            sys.exit("You can't default basis number (_basis_number) if a basis is prompted (_basis_prompt)")
        if not _basis == "":
            sys.exit("You can't default basis number (_basis_number) if a basis is specified (_basis)")
        if num_of_bases < _basis_number:
            sys.exit("You have chosen basis number " + str(_basis_number) + " but there are only " + str(num_of_bases) + " bases.")

        _basis = re.findall(r'VALUE=(.*)', bases)[_basis_number - 1]
        _basis = _basis.split("\"")[1]
        print("Chosen basis: " + str(_basis))



    #get the necessary hidden values again
    hidden_values = re.findall(r'hidden(.*)', form.text)
    num_of_values = len(hidden_values)
    value_names = [""] * num_of_values
    value_values = [""] * num_of_values
    
    for i in range(0, len(hidden_values)):
        value_names[i] = hidden_values[i].split('"')[2]
        value_values[i] = hidden_values[i].split('"')[4]
    
    #get the necessary hidden values again
    parameters = {}
    for i in range(0, len(hidden_values)):
        parameters[str(value_names[i])] = str(value_values[i])

    #add the basis to the parameters
    parameters['basisselect'] = _basis
    
    #set the remaining parameters to the default values
    parameters['basis11'] = '1'
    parameters['basis12'] = '0'
    parameters['basis13'] = '0'
    parameters['basis21'] = '0'
    parameters['basis22'] = '1'
    parameters['basis23'] = '0'
    parameters['basis31'] = '0'
    parameters['basis32'] = '0'
    parameters['basis33'] = '1'
    parameters['chooseorigin'] = 'false'
    parameters['dmax'] = '1'
    parameters['includestrain'] = 'true'
    parameters['inputbasis'] = 'list'
    parameters['origin1'] = '0'
    parameters['origin2'] = '0'
    parameters['origin3'] = '0'
    parameters['trynearest'] = 'true'
        
    #submit the distort input to the form
    form = requests.post(form_php, data = parameters)
    
    if re.search("Wyckoff positions in parent are not compatible with Wyckoff positions in subgroup", form.text):
        sys.exit("ISODISTORT: Wyckoff positions in parent are not compatible with Wyckoff positions in subgroup")
    
    if re.search("Subgroup and basis vectors are not compatible with parent.", form.text):
        sys.exit("ISODISTORT: Subgroup and basis vectors are not compatible with parent.")

    if re.search("This program has bombed", form.text): 
        sys.exit("ISODISTORT: This program has bombed")
        
    #amplitudes can already be found in this html. No need for mode display submission
    return form.text

#extract the amplitude of a certain mode and irreducible represenation from the final html file. 
def extract_amplitude(_out_html, _irrep, _disp, _abs = True):
    #escape brackets in the displacements
    _disp = _disp.replace('(', '\\(')
    _disp = _disp.replace(')', '\\)')
    _disp = _disp.replace('[', '\\[')
    _disp = _disp.replace(']', '\\]')
    #escape regex metachars
    _irrep = _irrep.replace('+', '\\+')
    _irrep = _irrep.replace('-', '\\-')
    _irrep = _irrep.replace('*', '\\*')
    _disp = _disp.replace('+', '\\+')
    _disp = _disp.replace('-', '\\-')
    _disp = _disp.replace('*', '\\*')
    #search for the specified irrep
    irrep_pattern = '' + _irrep + '\\s' + '(.|\\n)*?<p>'
    irrep_data = re.search(irrep_pattern, _out_html)
    amplitude = 0
    if irrep_data == None:
        print("Warning: irrep not present. Setting amplitude to 0.")
    else:
        #search the displacements in the irrep data
        disp_pattern = '(.*)' + _disp
        disp_data = re.search(disp_pattern, irrep_data[0])
        if disp_data == None:
            print("Warning: displacement not present. Setting amplitude to 0.")
        else:
            amplitude = float(disp_data[0].split("\"")[5])
    if _abs:
        amplitude = abs(amplitude)
    return amplitude   
    
#_____________________________________________________________________________________________#

def find_supercell(prim_cell, super_cell):
    
    '''
    Given two paths to a structure file, returns the transformation M from 
    primitive to supercell and the corresponding reciprocal transformation
    matrix. For more info see:
    math.stackexchange.com/questions/1078157/components-of-vector-in-dual-basis-transform-covariantly
    
    and
    
    github.com/QijingZheng/VaspBandUnfolding/blob/master/unfold.py
    
    Let M the transformation matrix between supercell and primitive cell,
    and A the column wise supercell vectors and a the column wise primitive
    cell vectors. Then M is given as:
            M = np.dot(A, np.linalg.inv(a)) 
    
    In real space, the basis vectors of Supercell (A) and those of the
    primitive cell (a) satisfy:
            A = np.dot(M, a);      a = np.dot(np.linalg.inv(M), A)
    
    Whereas in reciprocal space
            b = np.dot(M.T, B);    B = np.dot(np.linalg.inv(M).T, b)            
    
    hence to transform a given kpt in rel coords from b -> B:
            
            K_SC = np.dot(k_PC, M.T)
            
    uses pymatgen to load structure files and match the structures
            
    '''
    
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    
    PC = Structure.from_file(prim_cell)
    SC = Structure.from_file(super_cell)
    
    sm_matchter = StructureMatcher(primitive_cell=False, attempt_supercell=True)
    
    M, shift, match_atoms = sm.get_transformation(SC, PC)
    
    k_trans = np.linalg.inv(M).T
    
    return trans, k_trans, shift



def K_from_k(k, M):
    '''
    Get the K vector of the supercell onto which the k vector of the primitive
    cell folds.
    '''
    M = np.array(M)
    Kc = np.dot(k, M.T)
    return Kc