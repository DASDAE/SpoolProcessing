import numpy as np
import dascore as dc
import os
from tqdm.auto import tqdm
from glob import glob
import re


def get_sp_length(sp):
    sp_df = sp.get_contents()
    sp_length = (sp_df["time_max"].max() - sp_df["time_min"].min()) / np.timedelta64(
        1, "s"
    )
    return sp_length


def sp_process(
    sp,
    output_path,
    process_fun,
    pre_process=None,
    post_process=None,
    patch_size=1,
    overlap=None,
    save_file_size=200,
    merge_tolerance=3,
    overwrite=True,
    **kargs,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if overwrite:
        files = glob(output_path + "/*.h5")
        for file in files:
            os.remove(file)
        file = output_path + "/.dascore_index.h5"
        if os.path.isfile(file):
            os.remove(file)

    cont_sp = sp.chunk(
        time=None, tolerance=merge_tolerance
    )  # merge patches into continuous spools
    print("Found {} continuous datasets".format(len(cont_sp)))

    for i, cont_info in tqdm(cont_sp.get_contents().iterrows(), desc="Spool Loop"):
        csp = sp.select(time=(cont_info["time_min"], cont_info["time_max"]))
        sp_length = get_sp_length(csp)
        if sp_length > patch_size:
            sp_chunk = csp.chunk(
                time=patch_size,
                overlap=overlap,
                tolerance=merge_tolerance,
                keep_partial=True,
            )
        else:
            sp_chunk = csp.chunk(time=None, tolerance=merge_tolerance)
        sp_output = []
        sp_size = 0
        for patch in tqdm(sp_chunk, desc="Patch Loop", leave=False):
            if pre_process is not None:
                patch = pre_process(patch, **kargs)

            pro_patch = process_fun(patch, **kargs)

            if post_process is not None:
                pro_patch = post_process(pro_patch, **kargs)

            sp_output.append(pro_patch)
            sp_size += pro_patch.data.nbytes / 1.0e6
            if sp_size > save_file_size:
                output_spool(sp_output, output_path)
                sp_output = []
                sp_size = 0

        if len(sp_output) > 0:
            output_spool(sp_output, output_path)

    print("processing succeeded")
    return sp_output


def merge_patch_list(sp_output):
    # simple one, will discontinue once the bug in chunk is fixed.
    data = []
    taxis = []
    for patch in sp_output:
        data.append(patch.data)
        taxis.append(patch.coords["time"])

    taxis = np.concatenate(taxis)
    ind = np.argsort(taxis)
    taxis = taxis[ind]

    if patch.dims[0] == "time":
        data = np.vstack(data)
        data = data[ind, :]
        coords = {"time": taxis, "distance": patch.coords["distance"]}
    else:
        data = np.hstack(data)
        data = data[:, ind]
        coords = {"distance": patch.coords["distance"], "time": taxis}

    merged_patch = patch.new(data=data, coords=coords)
    return merged_patch


def output_spool(sp_output, output_path):
    patch_output = merge_patch_list(sp_output)
    #     patch_output = dc.spool(sp_output).chunk(time=None)[0]
    output_filename = os.path.join(output_path, get_filename(patch_output))
    dc.write(patch_output, output_filename, "dasdae")
    return output_filename


def get_filename(patch, time_string_length=19):
    bgstr = str(patch.attrs["time_min"])[:time_string_length]
    edstr = str(patch.attrs["time_max"])[:time_string_length]
    filename = bgstr + "_to_" + edstr + ".h5"
    filename = clean_filename(filename)
    return filename


def clean_filename(filename):
    # Remove illegal characters
    cleaned_filename = re.sub(r'[\/:*?"<>|]', "_", filename)

    # Remove leading/trailing spaces and dots
    cleaned_filename = cleaned_filename.strip(". ").replace(" ", "_")

    return cleaned_filename


def get_edge_effect_time(dt, fun, total_T, tol=1e-3, **kargs):
    N = int(total_T / dt)

    taxis = (np.arange(N) - N // 2) * dt
    data = np.zeros_like(taxis)
    data[N // 2] = 1

    coords = {"time": taxis, "distance": [0]}
    data = data.reshape((-1, 1))
    attrs = {"d_time": dt, "d_distance": 1}

    new_data = dc.Patch(
        data=data, coords=coords, dims=["time", "distance"], attrs=attrs
    )
    process_data = new_data.pipe(fun, **kargs)

    data = process_data.data[:, 0]

    max_val = np.max(np.abs(data))
    ind = np.abs(data) > max_val * tol
    ind_1 = np.where(ind)[0][0]
    ind_2 = np.where(ind)[0][-1]

    new_taxis = process_data.coords["time"]
    new_taxis = (new_taxis - new_taxis[0]) / np.timedelta64(1, "s") - N // 2 * dt

    edge_t = max(np.abs(new_taxis[ind_1]), np.abs(new_taxis[ind_2]))

    if edge_t * 2 >= total_T:
        raise ValueError(
            f"edge_t {edge_t} s is equal or larger than half of\
            the processing chunk size {total_T} s.\
            Please increase memory_size or tolerance."
        )

    return edge_t


def get_chunk_time(
    memory_size,
    sampling_rate,
    num_ch,
    bytes_per_element=8,
    processing_factor=5,
    memory_safety_factor=1.2,
):
    mem_size_per_second = (
        sampling_rate
        * num_ch
        * bytes_per_element
        * processing_factor
        * memory_safety_factor
        / 1e6
    )  # in MB
    patch_length = memory_size / mem_size_per_second  # in sec
    return patch_length
