"""
IMPORTANT NOTE:
This file makes use of a database that is no longer running and thus cannot be run as-is.
Nonetheless, the best way to understand the processing pipelne is to see how it was run originally.

These functions would download connectivity information and get a list of
all Chandelier cells, pyramidal cells, the AIS limit markers, and synapses. Where needed, it downloads new
pyramidal cell meshes and extracts the AIS, uses this AIS definition to determine which input synapses were
associated with the AIS. Finally, it takes the synapse information and packs it up into dataframes for
analysis.

If one were interested in replicating the pipeline, please follow the logic to the actual analysis steps
rather than the early downloading stages.

Please reach out with any questions at caseys@alleninstitute.org
"""


import os
import sys
from pathlib import Path

base_dir = str(Path("..").resolve())
mesh_dir = base_dir + "/data/meshes"
skel_dir = base_dir + "/data/skeletons/"
sys.path.append(base_dir + "/src")

import ais_pipeline_scripting_tools as utils

import tqdm.autonotebook as tqdm
import pandas as pd
import numpy as np
# from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
from meshparty import trimesh_io, skeleton_io
from multiwrapper import multiprocessing_utils as mu

mesh_cv_path =  "precomputed://gs://microns_public_datasets/pinky100_v185/seg"
voxel_resolution = [3.58, 3.58, 40]
voxel_scaling = [0.895, 0.895, 1]

data_version = 185
ais_table = 'ais_bounds_v3'
soma_table = 'soma_valence_v2'
data_filename_scheme = "/in/data_v{data_version}_{ais_table}_{chc_table}.h5"

matrix_filename = base_dir + "/data/in/pinky_rotation.npy"

synapse_table_default = 'pni_synapses_i3'
dataset_name_default = 'pinky100'

def load_rotation_matrix(filename=matrix_filename):
    Rtrans = np.load(matrix_filename)
    return Rtrans


def compute_ais_synapse_data(
    chc_table,
    ais_table,
    data_version,
    ais_bound_table="ais_bounds_v2",
    synapse_table=synapse_table_default,
    dataset_name=dataset_name_default,
    max_dist_to_mesh=75,
    save_ais_mesh=True,
    save_soma_mesh=True,
    soma_table=soma_table,
    soma_radius=15000,
    n_threads=None,
    redownload_meshes=False,
    rotate_ais_points=True,
):
    print(f"Voxel scaling: {voxel_scaling}")
    print(f"Voxel resolution: {voxel_resolution}")

    if n_threads is not None:
        n_threads = int(n_threads)
    else:
        n_threads = mu.cpu_count()

    dl = AnalysisDataLink(
        dataset_name=dataset_name,
        sqlalchemy_database_uri=sql_database_uri_base,
        materialization_version=data_version,
        verbose=False,
    )

    # Get ChC root ids
    print("\tQuerying chandelier cells...")
    chc_df = dl.query_cell_ids(chc_table)

    # Get AIS root ids and bounds
    print("\tQuerying AIS data...")
    ais_df = dl.query_cell_ids(ais_table)
    complete_ais_ids = np.unique(ais_df.pt_root_id)
    ais_bounds_df = dl.query_cell_ids(ais_bound_table)

    if redownload_meshes:
        print("\tPRE-DOWNLOADING ALL MESHES!")
        _download_all_meshes(complete_ais_ids, mesh_cv_path, mesh_dir, n_threads)
        print("\tContinuing processing...")

    # Get all postsynaptic synapses onto AIS neurons
    if save_soma_mesh:
        soma_df = dl.query_cell_ids(soma_table)
    else:
        soma_df = None

    print("\tQuerying synapses...")
    input_syn_df = dl.query_synapses(synapse_table, post_ids=complete_ais_ids)
    print(len(input_syn_df))
    print("\tProcessing AISes...")
    ais_synapse_data, oids_failed_ais, zero_synapse_oids = _compute_ais_synapse_data(
        complete_ais_ids,
        mesh_cv_path,
        mesh_dir,
        max_dist_to_mesh,
        ais_bounds_df,
        chc_df,
        save_ais_mesh,
        save_soma_mesh=save_soma_mesh,
        input_syn_df=input_syn_df,
        soma_df=soma_df,
        soma_radius=soma_radius,
        n_threads=n_threads,
        rotate_ais_points=rotate_ais_points,
    )
    return ais_synapse_data, oids_failed_ais, zero_synapse_oids, complete_ais_ids


def _download_all_meshes(complete_ais_ids, mesh_cv_path, mesh_dir, n_threads):
    if n_threads is None:
        n_threads = mu.cpu_count()
    trimesh_io.download_meshes(
        complete_ais_ids,
        mesh_dir,
        mesh_cv_path,
        overwrite=False,
        n_threads=n_threads,
        verbose=True,
    )


def _compute_ais_synapse_data(
    complete_ais_ids,
    mesh_cv_path,
    mesh_dir,
    max_dist_to_mesh,
    ais_bounds_df,
    chc_df,
    save_ais_mesh,
    save_soma_mesh,
    input_syn_df,
    soma_df,
    soma_radius,
    n_threads,
    rotate_ais_points,
):

    if save_ais_mesh is True:
        ais_mesh_dir = mesh_dir + "/ais_meshes"
        if not os.path.exists(ais_mesh_dir):
            os.mkdir(ais_mesh_dir)
    else:
        ais_mesh_dir = None

    if rotate_ais_points is True:
        Rtrans = load_rotation_matrix()
    else:
        Rtrans = None

    ais_data_dfs = {}
    oids_failed_ais = []
    zero_synapse_oids = []
    print("\tComputing AIS meshes and synapses...")
    multi_args = _format_multi_args(
        complete_ais_ids,
        ais_bounds_df,
        soma_df,
        input_syn_df,
        mesh_cv_path,
        save_ais_mesh,
        save_soma_mesh,
        voxel_resolution,
        soma_radius,
        mesh_dir,
        max_dist_to_mesh,
        Rtrans,
    )
    if n_threads > 1:
        ais_data_results = mu.multiprocess_func(
            _compute_single_ais, multi_args, verbose=True, n_threads=n_threads
        )
    else:
        ais_data_results = []
        pbar = tqdm.tqdm(multi_args)
        for arg in pbar:
            pbar.set_description("AIS {}".format(arg[0]))
            ais_data_results.append(_compute_single_ais(arg))

    for result, oid in zip(ais_data_results, complete_ais_ids):
        if type(result) is pd.DataFrame:
            ais_data_dfs[oid] = result
        elif result == "failed_synapses":
            zero_synapse_oids.append(oid)
        else:
            oids_failed_ais.append(oid)

    consensus_chc_df = (
        chc_df[["pt_root_id", "func_id"]].groupby("pt_root_id").mean().reset_index()
    )
    consensus_chc_df.rename(columns={"func_id": "is_chandelier"}, inplace=True)
    consensus_chc_df["is_chandelier"] = consensus_chc_df["is_chandelier"] == 1

    if len(ais_data_results) > 0:
        ais_synapse_data = pd.concat([x for x in ais_data_dfs.values()]).reset_index(
            drop=True
        )
        ais_synapse_data = ais_synapse_data.merge(
            consensus_chc_df,
            left_on="pre_pt_root_id",
            right_on="pt_root_id",
            how="left",
        ).drop(columns="pt_root_id")
        ais_synapse_data["is_chandelier"] = ais_synapse_data["is_chandelier"] == True
    else:
        ais_synapse_data = pd.DataFrame()
    return ais_synapse_data, oids_failed_ais, zero_synapse_oids


def aggregate_ais_dataframes(complete_ais_ids, ais_synapse_data):
    print("\tComputing aggregate AIS information")
    aggregated_ais_chc_syn_df = utils.aggregate_postsynaptic(
        complete_ais_ids,
        ais_synapse_data[ais_synapse_data["is_chandelier"] == True],
        return_size=True,
    )
    aggregated_ais_chc_syn_df = utils.compute_per_ais_properties(
        aggregated_ais_chc_syn_df
    )

    aggregated_ais_non_syn_df = utils.aggregate_postsynaptic(
        complete_ais_ids,
        ais_synapse_data[ais_synapse_data["is_chandelier"] == False],
        return_size=True,
    )
    aggregated_ais_non_syn_df = utils.compute_per_ais_properties(
        aggregated_ais_non_syn_df
    )

    aggregated_ais_syn_df = aggregated_ais_chc_syn_df.merge(
        aggregated_ais_non_syn_df, on="post_pt_root_id", suffixes=["_chc", "_non"]
    )

    ais_len_df = (
        ais_synapse_data[["post_pt_root_id", "ais_len"]]
        .groupby("post_pt_root_id")
        .agg(np.unique)
    )
    aggregated_ais_syn_df = aggregated_ais_syn_df.merge(
        ais_len_df, left_on="post_pt_root_id", right_index=True
    )
    if "n_syn_soma" in ais_synapse_data.columns:
        n_syn_soma_df = (
            ais_synapse_data[["post_pt_root_id", "n_syn_soma"]]
            .groupby("post_pt_root_id")
            .agg(np.unique)
            .reset_index()
        )
        aggregated_ais_syn_df = aggregated_ais_syn_df.merge(
            n_syn_soma_df, on="post_pt_root_id"
        )
    return aggregated_ais_syn_df


def save_ais_synapse_data(
    ais_synapse_data,
    aggregated_ais_syn_df,
    oids_failed_ais,
    zero_synapse_oids,
    chc_table,
    ais_table,
    data_version,
    params,
    data_dir,
):
    info_dict = {
        "data_version": data_version,
        "ais_table": ais_table,
        "chc_table": chc_table,
    }
    data_filename = data_dir + data_filename_scheme.format_map(info_dict)
    print("\tSaving data to {}".format(data_filename))
    ais_synapse_data.to_hdf(data_filename, "ais_synapse_data", mode="a")

    aggregated_ais_syn_df.to_hdf(data_filename, "aggregated_ais_syn_df", mode="a")

    oids_failed_ais = pd.Series(oids_failed_ais)
    oids_failed_ais.to_hdf(data_filename, "oids_failed_ais", mode="a")

    zero_synapse_oids = pd.Series(zero_synapse_oids)
    zero_synapse_oids.to_hdf(data_filename, "zero_synapse_oids", mode="a")

    param_df = pd.Series(params)
    param_df.to_hdf(data_filename, "parameters", mode="a")


def load_ais_synapse_data(chc_table, ais_table, data_version, data_dir):
    info_dict = {
        "data_version": data_version,
        "ais_table": ais_table,
        "chc_table": chc_table,
    }
    data_filename = data_dir + data_filename_scheme.format_map(info_dict)

    if os.path.isfile(data_filename):
        print("Loading data from {}".format(data_filename))
        ais_synapse_data = pd.read_hdf(data_filename, "ais_synapse_data")
        aggregated_ais_syn_df = pd.read_hdf(data_filename, "aggregated_ais_syn_df")
        oids_failed_ais = pd.read_hdf(data_filename, "oids_failed_ais").values
        zero_synapse_oids = pd.read_hdf(data_filename, "zero_synapse_oids").values
        return (
            ais_synapse_data,
            aggregated_ais_syn_df,
            oids_failed_ais,
            zero_synapse_oids,
        )
    else:
        print("ais_synapse_data is not computed for these parameters!")
        return None, None, None, None


def _format_multi_args(
    oids,
    ais_bounds_df,
    soma_df,
    input_syn_df,
    mesh_cv_path,
    save_ais_mesh,
    save_soma_mesh,
    voxel_resolution,
    soma_radius,
    mesh_dir,
    max_dist_to_mesh,
    Rtrans,
):
    multi_args = []
    for oid in oids:
        multi_args.append(
            [
                oid,
                ais_bounds_df,
                soma_df,
                input_syn_df,
                mesh_cv_path,
                save_ais_mesh,
                save_soma_mesh,
                voxel_resolution,
                soma_radius,
                mesh_dir,
                max_dist_to_mesh,
                Rtrans,
            ]
        )
    return multi_args


def _compute_single_ais(data):
    (
        oid,
        ais_bounds_df,
        soma_df,
        input_syn_df,
        mesh_cv_path,
        save_ais_mesh,
        save_soma_mesh,
        voxel_resolution,
        soma_radius,
        mesh_dir,
        max_dist_to_mesh,
        Rtrans,
    ) = data

    mm = trimesh_io.MeshMeta(
        disk_cache_path=mesh_dir,
        cache_size=0,
        cv_path=mesh_cv_path,
        map_gs_to_https=True,
        voxel_scaling=voxel_scaling,
    )

    if save_ais_mesh is True:
        ais_mesh_dir = mesh_dir + "/ais_meshes"
    else:
        ais_mesh_dir = None

    bnds_cell_df = ais_bounds_df[ais_bounds_df["pt_root_id"] == oid]
    if not (any(bnds_cell_df["func_id"] == 0) and any(bnds_cell_df["func_id"] == 1)):
        return "failed_ais_bounds"

    best_zero_pt = bnds_cell_df[bnds_cell_df["func_id"] == 0].iloc[
        np.argmax(bnds_cell_df[bnds_cell_df["func_id"] == 0]["id"].values)
    ]["pt_position"]
    best_one_pt = bnds_cell_df[bnds_cell_df["func_id"] == 1].iloc[
        np.argmax(bnds_cell_df[bnds_cell_df["func_id"] == 1]["id"].values)
    ]["pt_position"]
    ais_point_list_vxl = [best_zero_pt, best_one_pt]

    ais_points = utils.scale_voxel_to_euclidean(
        np.vstack(ais_point_list_vxl), voxel_res=voxel_resolution
    )
    nrn_syn_df = input_syn_df.query(f"post_pt_root_id=={oid}")

    if len(nrn_syn_df) == 0:
        return "failed_synapses"

    mesh = mm.mesh(seg_id=oid)

    if save_soma_mesh:
        if len(soma_df[soma_df["pt_root_id"] == oid]) > 0:
            center_point = (
                np.mean(soma_df[soma_df["pt_root_id"] == oid]["pt_position"].values)
                * voxel_resolution
            )
        else:
            center_point = None
    else:
        center_point = None

    fix_filename = mesh_dir + "/{}.h5".format(oid)
    mesh_clean, minds_ais = utils.clean_and_repair_ais(
        mesh,
        ais_points,
        soma_point=center_point,
        soma_radius=soma_radius,
        line_dist_th=15000,
        comp_size_th=500,
        mesh_distance_upper_bound=250,
        do_save_on_repair=True,
        filename=fix_filename,
    )
    if mesh_clean is None:
        return "failed_clean"

    is_ais = utils.compute_ais(mesh_clean, minds_ais, d_pad=1000)
    if is_ais is None:
        return "failed_ais_finding"

    syn_mesh_inds = utils.attach_synapses_to_mesh(
        mesh_clean, nrn_syn_df, d_th=150, return_unmasked=True
    )
    syn_mesh_inds_keep = syn_mesh_inds >= 0

    is_ais_syn = np.full(syn_mesh_inds.shape, False)
    is_ais_syn[syn_mesh_inds_keep] = is_ais[syn_mesh_inds[syn_mesh_inds_keep]]
    ais_syn_mesh_inds = syn_mesh_inds[is_ais_syn]

    mesh_ais = mesh.apply_mask(is_ais)
    ais_syn_mesh_inds_small = mesh_ais.filter_unmasked_indices(ais_syn_mesh_inds)
    if save_ais_mesh:
        filename = f"{ais_mesh_dir}/{oid}_ais.h5"
        mesh_ais.write_to_file(filename)
    if save_soma_mesh:
        mesh_soma = utils.compute_soma(
            mesh_clean,
            center_point,
            soma_radius,
            mesh_ais,
            save_mesh=True,
            filename=f"{ais_mesh_dir}/{oid}_soma.h5",
        )
        if mesh_soma is not None:
            n_syn_soma = sum(mesh_soma.node_mask[syn_mesh_inds[syn_mesh_inds_keep]])
            nrn_syn_df["n_syn_soma"] = n_syn_soma
        else:
            nrn_syn_df["n_syn_soma"] = np.nan

    ais_syn_df = nrn_syn_df[is_ais_syn]
    ais_syn_df["post_pt_mesh_ind"] = ais_syn_mesh_inds

    ais_sk = utils.skeletonize_ais(mesh_ais)
    skeleton_io.write_skeleton_h5(ais_sk, skel_dir + "/sk_ais_{}.h5".format(oid))
    print(f"\t\tComputing properties for {oid}...")
    if len(ais_syn_df) > 0:
        try:
            ais_data_df = utils.append_spatial_data(ais_syn_df, mesh_ais, ais_points)

            sk_distance, ais_len = utils.distance_along_skeleton(ais_sk, ais_syn_df)
            ais_data_df["d_top_skel"] = sk_distance
            ais_data_df["ais_len"] = ais_len[0]

            orientations = utils.point_orientation_from_zx_slice(
                ais_syn_mesh_inds_small,
                mesh_ais,
                max_dist_to_mesh,
                rotation_matrix=Rtrans,
            )
            ais_data_df["orientation"] = orientations
            ais_data_df["syn_per_edge"] = utils.synapses_in_edge(ais_data_df)
            ais_data_df = utils.clean_df(ais_data_df)
        except:
            print(f"Failed on {oid}")
            raise Exception

    return ais_data_df
