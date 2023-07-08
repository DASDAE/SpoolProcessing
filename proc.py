import numpy as np
import dascore as dc
import os
from tqdm import tqdm
from glob import glob

from .utils import sp_process,get_edge_effect_time

def _std_processing(DASdata,**kargs):

#     DASdata = DASdata.tran.velocity_to_strain_rate()
    axis = DASdata.dims.index('time')
    newdata = np.std(DASdata.data,axis=axis)

    daxis = DASdata.coords['distance']

    bgtime = DASdata.attrs['time_min']
    edtime = DASdata.attrs['time_max']
    d_time = edtime-bgtime
    taxis = np.array([bgtime+(edtime-bgtime)/2])

    newdata = newdata.reshape((-1,1))
    coords = {'distance':daxis,'time':taxis}
    dims = ['distance','time']
    out_data = DASdata.new(data=newdata,coords=coords,dims=dims)
    out_data = out_data.update_attrs(d_time=d_time)

    return out_data


def std(sp,output_path,**kargs):
    """
    Downsample the spool by calculating the standard deviation value over a patch size

    Parameters:
    - sp: input dascore spool
    - output_path (str): The folder where the processed output will be saved.
    
    Keywords:
    - patch_size=1: number of seconds std is calculated at each channel
    - over_write=True: whether overwrite existing files in the folder
    - pre_process=None: pre_process function applied to data before std calculation
    - **kargs: Additional keyword arguments to be passed to the 'sp_process' function.

    Returns:
    - The result of the 'sp_process' function


    Example usage:
    sp = dc.get_example_spool()
    def fun(patch,**kargs):   
        # **kargs is important to allow the framework to pass keywords to pre and post processing function
        return patch.tran.velocity_to_strain_rate()
    output = std(sp, "output_folder", patch_size=0.5, overwrite=True, pre_process=fun)
    """
    
    return sp_process(sp,output_path,_std_processing, **kargs)


def _down_sample_processing(patch,freq=5,nqfreq_ratio=0.8, **kargs):
    dt = np.timedelta64(int(1/freq*1e9),'ns')
    corner_f = freq*0.5*nqfreq_ratio
    
    proc_patch = patch
    
    proc_patch = proc_patch.pass_filter(time=(None,corner_f))
    new_taxis = np.arange(patch.attrs['time_min'],
                         patch.attrs['time_max'],
                         dt)
    proc_patch = proc_patch.interpolate(time=new_taxis)
    
#     proc_patch = proc_patch.resample(time=dt)
    
    return proc_patch


def down_sample(sp,output_path,freq,edge_time_ratio=1.2, memory_size=1000, **kargs):
    """
    Downsample the spool by calculating the standard deviation value over a patch size

    Parameters:
    - sp: input dascore spool
    - output_path (str): The folder where the processed output will be saved.
    - freq: new sampling rate in Hz
    
    Keywords:
    - memory_size = 1000: processing chunk size in MB
    - nqfreq_ratio = 0.8: ratio of low-pass filter corner frequency to Nquist frequency
    - edge_time_ratio = 1.2: overlap window multiplier
    - over_write=True: whether overwrite existing files in the folder
    - pre_process=None: pre_process function applied to data before std calculation
    - **kargs: Additional keyword arguments to be passed to the 'sp_process' function.

    Returns:
    - True indicating successful processing


    Example usage:
    sp = dc.get_example_spool()
    def fun(patch):
        return patch.tran.velocity_to_strain_rate()
    output = down_sample(sp, "output_folder", 0.5, overwrite=True, pre_process=fun, memory_size=1000)
    """

    dt = sp.get_contents()['d_time'].iloc[0]/np.timedelta64(1,'s')
    edge_time = get_edge_effect_time(dt,_down_sample_processing,1/freq*100,freq=freq)

    item = sp.get_contents().iloc[0]
    chanN = int((item['distance_max']-item['distance_min'])/item['d_distance'])

    mem_size_per_second = 1/dt*chanN*4/1e6
    chunk_size = memory_size/mem_size_per_second
    overlap = edge_time*edge_time_ratio*2

    sp_df = sp.get_contents()
    sp_length = (sp_df['time_max'].max()-sp_df['time_min'].min())/np.timedelta64(1,'s')

    print(f'Chunk size: {chunk_size} s, Overlap: {overlap} s')

    if chunk_size>sp_length:
        raise ValueError(f'Chunk size {chunk_size} s is larger than spool length {sp_length} s.  Please decrease memory_size or use Patch process ')

    if overlap*2 > chunk_size:
        raise ValueError(f'Overlap {overlap*2} s is larger than chunk size. Please increase memory_size.')

    if overlap*2 > 0.5*chunk_size:
        print(f'Overlap {overlap*2} is larger than 50% of chunk size. It is inefficient. Please increase memory_size.')

    def post_proc(patch,edge_win=None,**kargs):
        edge_time = np.timedelta64(int(edge_win*1e9),'ns')
        bgtime = patch.attrs['time_min']+edge_time
        edtime = patch.attrs['time_max']-edge_time
        return patch.select(time=(bgtime,edtime))

    sp_output = sp_process(sp,'./test',_down_sample_processing,freq=freq,
                                 post_process=post_proc,
                                patch_size=chunk_size,overlap=overlap,edge_win=overlap/2,**kargs)


    return sp_output
    