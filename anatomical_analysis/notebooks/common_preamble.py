import os
import sys
from pathlib import Path

base_dir = str(Path("..").resolve())
mesh_dir = base_dir + "/data/meshes"
skel_dir = base_dir + "/data/skeletons"
ais_mesh_dir = mesh_dir + "/ais_meshes"

plot_dir = base_dir + "/plots"

sys.path.append(base_dir + "/src")
from ais_synapse_utils import load_ais_synapse_data

import tqdm as tqdm
import pandas as pd
import numpy as np
from scipy import spatial, sparse

import vtk
import matplotlib as mpl

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

from meshparty import trimesh_io, trimesh_vtk, skeleton_io

from dotenv import load_dotenv

data_version = 185
ais_table = "ais_bounds_v3"  # manual_ais for core, ais_bounds for all
chc_table = "is_chandelier_v5"  # consult materialization for the best version

mesh_cv_path = "precomputed://gs://microns_public_datasets/pinky100_v185/seg"
voxel_resolution = [3.58, 3.58, 40]
voxel_scaling = [0.895, 0.895, 1]

(
    ais_synapse_data_all,
    aggregated_ais_syn_df_all,
    oids_failed,
    zero_syn_oids,
) = load_ais_synapse_data(chc_table, ais_table, data_version, base_dir + "/data")
print("\n")
if oids_failed is not None:
    if len(oids_failed) > 0:
        print(
            "Warning: {} oids failed on AIS synapse computation".format(
                len(oids_failed)
            )
        )
    if len(zero_syn_oids) > 0:
        print("Warning: some oids had no AIS synapses")

proofreading_fname = f"{base_dir}/data/in/synapse_proofreading.csv"
proofreading_df = pd.read_csv(proofreading_fname)
ais_synapse_data_all = ais_synapse_data_all.merge(
    proofreading_df[["syn_id", "synapse_correct"]],
    how="left",
    left_on="id",
    right_on="syn_id",
)
ais_synapse_data = (
    ais_synapse_data_all.query("synapse_correct==True")
    .drop(columns=["syn_id", "synapse_correct"])
    .reset_index()
)

plot_dir = base_dir + "/plots/v{}".format(data_version)
if not os.path.exists(plot_dir):
    print("Making new plot directory...")
    os.mkdir(plot_dir)
    os.mkdir(plot_dir + "/ais_views")

absolute_min_ais_len = 35000

ais_lens = np.unique(ais_synapse_data["ais_len"])
min_ais_len = np.min(ais_lens[ais_lens > absolute_min_ais_len])
print(f"Minimum AIS length: {min_ais_len}")


# dl = AnalysisDataLink(dataset_name=dataset_name,
#                       sqlalchemy_database_uri=sql_database_uri_base,
#                       materialization_version=data_version,
#                       verbose=False)

# chc_df = dl.query_cell_ids(chc_table)
# chc_ids = np.unique(chc_df[chc_df.func_id==1].pt_root_id)

# ais_df = dl.query_cell_ids(ais_table)
# complete_ais_ids = np.unique(ais_df.pt_root_id)

# from ais_synapse_utils import aggregate_ais_dataframes
# aggregated_ais_syn_df = aggregate_ais_dataframes(complete_ais_ids, ais_synapse_data).reset_index()

# # Use for the moment... probably need to fix when we explicitly set ids to ignore.
# from analyzable_soma_ids import oids_to_ignore
# oids_ignore = oids_to_ignore(data_version)

# ais_id_to_analyze = set(complete_ais_ids.astype(int)).difference(zero_syn_oids).difference(oids_failed).difference(oids_ignore)
# complete_ais_ids = np.array(list(ais_id_to_analyze))
# analyzable_soma_ids = np.array(list(set(complete_ais_ids).difference(oids_ignore)))

from paper_styles import *

set_rc_params(mpl)

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
