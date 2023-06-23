import numpy as np
import matplotlib.pyplot as plt
import dascore as dc
import os
from tqdm.auto import tqdm
from glob import glob

def sp_process(sp, output_path, process_fun, pre_process=None,
               patch_size=1, overlap=None, save_file_size=200, 
               overwrite=True, **kargs):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if overwrite:
        files = glob(output_path+'/*.h5')
        for file in files:
            os.remove(file)
        file = output_path+'/.dascore_index.h5'
        if os.path.isfile(file):
            os.remove(file)
        
    cont_sp = sp.chunk(time=None) # merge patches into continuous spools
    print('Found {} continuous datasets'.format(len(cont_sp)))
    
    for i,cont_info in tqdm(cont_sp.get_contents().iterrows(), desc='Spool Loop'):
        csp = sp.select(time=(cont_info['time_min'],cont_info['time_max']))
        sp_chunk = csp.chunk(time=patch_size, overlap=overlap, **kargs)
        sp_output = []
        sp_size = 0
        for patch in tqdm(sp_chunk, desc='Patch Loop', leave=False):
            if pre_process is not None:
                patch = pre_process(patch)
            pro_patch = process_fun(patch)
            sp_output.append(pro_patch)
            sp_size += pro_patch.data.nbytes/1.0e6
            if sp_size > save_file_size:
                output_spool(sp_output,output_path)
                sp_output = []
                sp_size=0

        if len(sp_output)>0:
            output_spool(sp_output,output_path)
       
    print('processing succeeded')
    return sp_output

def merge_patch_list(sp_output):
    # simple one, will discontinue once the bug in chunk is fixed. 
    data = []
    taxis = []
    for patch in sp_output:
        data.append(patch.data)
        taxis.append(patch.coords['time'])

    data = np.hstack(data)
    taxis = np.concatenate(taxis)

    ind = np.argsort(taxis)
    data = data[:,ind]
    taxis = taxis[ind]

    coords = {'distance':patch.coords['distance'],'time':taxis}

    merged_patch = patch.new(data=data,coords=coords)
    return merged_patch
            
def output_spool(sp_output, output_path):        
    print(len(sp_output))
    patch_output = merge_patch_list(sp_output)
#     patch_output = dc.spool(sp_output).chunk(time=None)[0]
    output_filename = os.path.join(output_path,get_filename(patch_output))
    dc.write(patch_output, output_filename,'dasdae')
    return output_filename
        
    
def get_filename(patch, time_string_length=19):
    bgstr = str(patch.attrs['time_min'])[:time_string_length]
    edstr = str(patch.attrs['time_max'])[:time_string_length]
    filename = bgstr+'__'+edstr+'.h5'
    return filename