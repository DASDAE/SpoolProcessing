"""
A Python script for different processings on a spool of DAS data.
"""

import numpy as np

from .utils import sp_process, get_edge_effect_time, get_chunk_time


def _std_processing(DASdata, **kargs):
    axis = DASdata.dims.index("time")
    if "pre_process" in kargs:
        DASdata = kargs["pre_process"](DASdata)

    newdata = np.std(DASdata.data, axis=axis)

    daxis = DASdata.coords["distance"]

    bgtime = DASdata.attrs["time_min"]
    edtime = DASdata.attrs["time_max"]
    d_time = edtime - bgtime
    taxis = np.array([bgtime + (edtime - bgtime) / 2])

    newdata = newdata.reshape((-1, 1))
    coords = {"distance": daxis, "time": taxis}
    dims = ["distance", "time"]
    out_data = DASdata.new(data=newdata, coords=coords, dims=dims)
    out_data = out_data.update_attrs(d_time=d_time)

    return out_data


def std(sp, output_path, **kargs):
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
        # **kargs is important to allow the framework
        to pass keywords to pre and post processing function
        return patch.tran.velocity_to_strain_rate()
    output = std(sp, "output_folder", patch_size=0.5, overwrite=True, pre_process=fun)
    """

    return sp_process(sp, output_path, _std_processing, **kargs)


def _low_freq_processing(patch, freq=5, nqfreq_ratio=0.8, **kargs):
    dt = np.timedelta64(int(1 / freq * 1e9), "ns")
    corner_f = freq * 0.5 * nqfreq_ratio

    proc_patch = patch
    proc_patch = proc_patch.pass_filter(time=(None, corner_f))

    new_t_axis = np.arange(patch.attrs["time_min"], patch.attrs["time_max"], dt)
    proc_patch = proc_patch.interpolate(time=new_t_axis)

    return proc_patch


def low_freq(sp, output_path, freq, edge_time_ratio=1.2, memory_size=1000, **kargs):
    """
    Low-pass filter the spool.

    Parameters:
    - sp: Input dascore spool
    - output_path (str): The directory path where the processed output will be saved
    - freq: Target sampling rate in Hz
    So the target corner frequency = Nyq_new = 0.5*freq

    Keywords:
    - memory_size = 1000: Available Memory in MB for low-freq processing
    - edge_time_ratio = 1.2: overlap window multiplier
    - pre_process = None: pre_process function applied to data before std calculation
    - **kargs: Additional keyword arguments to be passed to the 'sp_process' function

    Returns:
    - True indicating successful processing


    Example usage:
    sp = dc.get_example_spool()
    def fun(patch):
        return patch.tran.velocity_to_strain_rate()
    output = down_sample(
        sp, "output_folder", 0.5, overwrite=True,
        pre_process=fun, memory_size=1000
        )
    """

    item = sp.get_contents().iloc[0]
    num_ch = int((item["distance_max"] - item["distance_min"]) / item["d_distance"])

    dt = sp.get_contents()["d_time"].iloc[0] / np.timedelta64(1, "s")

    chunk_size = get_chunk_time(
        memory_size=memory_size, sampling_rate=dt, num_ch=num_ch
    )

    edge_time = get_edge_effect_time(dt, _low_freq_processing, chunk_size, freq=freq)

    overlap = edge_time * edge_time_ratio * 2

    print(f"Chunk size: {chunk_size} s, Overlap: {overlap} s")

    if overlap * 2 > chunk_size:
        raise ValueError(
            f"Overlap {overlap*2} s is larger than chunk size. \
            Please increase memory_size."
        )

    if overlap * 2 > 0.5 * chunk_size:
        print(
            f"Overlap {overlap*2} is larger than 50% of chunk size. \
            It is inefficient. Please increase memory_size."
        )

    def post_proc(patch, edge_win=None, **kargs):
        edge_time = np.timedelta64(int(edge_win * 1e9), "ns")
        bgtime = patch.attrs["time_min"] + edge_time
        edtime = patch.attrs["time_max"] - edge_time
        return patch.select(time=(bgtime, edtime))

    sp_output = sp_process(
        sp,
        output_path,
        _low_freq_processing,
        freq=freq,
        post_process=post_proc,
        patch_size=chunk_size,
        overlap=overlap,
        edge_win=overlap / 2,
        **kargs,
    )

    return sp_output
