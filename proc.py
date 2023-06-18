import numpy as np
import dascore as dc
import os
from tqdm import tqdm
from glob import glob

from .utils import sp_process

def _std_processing(DASdata):

    DASdata = DASdata.tran.velocity_to_strain_rate()
    axis = DASdata.dims.index('time')
    newdata = np.std(DASdata.data,axis=axis)

    daxis = DASdata.coords['distance']

    bgtime = DASdata.attrs['time_min']
    edtime = DASdata.attrs['time_max']
    d_time = edtime-bgtime
    taxis = np.array([bgtime+(edtime-bgtime)/2])

    newdata = newdata.reshape((-1,1))
    coords = {'distance':daxis,'time':taxis}
    out_data = DASdata.new(data=newdata,coords=coords)
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
    - **kargs: Additional keyword arguments to be passed to the 'sp_process' function.

    Returns:
    - The result of the 'sp_process' function


    Example usage:
    >>> sp = dc.get_example_spool()
    >>> output = std(sp, "output_folder", patch_size=0.5, overwrite=True)
    """
    
    return sp_process(sp,output_path,_std_processing, **kargs)
