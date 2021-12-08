"""
IMPORTANT NOTE:
This file makes use of a database that is no longer running and thus cannot be run as-is.
Nonetheless, the best way to understand the processing pipelne is to see how it was run originally.

This function, when run from the command line, would download connectivity information and get a list of
all Chandelier cells, pyramidal cells, the AIS limit markers, and synapses. Where needed, it downloads new
pyramidal cell meshes and extracts the AIS, uses this AIS definition to determine which input synapses were
associated with the AIS. Finally, it takes the synapse information and packs it up into dataframes for
analysis.
"""


import sys
from pathlib import Path

base_dir = str(Path("..").resolve())
data_dir = base_dir + "/data"
mesh_dir = base_dir + "/data/meshes"
sys.path.append(base_dir + "/src")
from ais_synapse_utils import (
    compute_ais_synapse_data,
    save_ais_synapse_data,
    aggregate_ais_dataframes,
)
import click


@click.command()
@click.option("-v", "--data_version", required=True)
@click.option("-c", "--chc_table", default="is_chandelier_v5", required=False)
@click.option("-a", "--ais_table", default="ais_bounds_v3", required=False)
@click.option("-b", "--bounds_table", default="ais_bounds_v3", required=False)
@click.option("-d", "--max_dist_to_mesh", default=75, required=False)
@click.option("-m", "--save_ais_mesh", default=True, required=False, is_flag=True)
@click.option("-o", "--save_soma_mesh", default=True, required=False, is_flag=True)
@click.option("-s", "--soma_table", default="soma_valence_v2", required=False)
@click.option("-r", "--soma_radius", default=15000, required=False)
@click.option("-n", "--n_threads", default=None, required=False)
@click.option("--redownload-meshes", default=False, required=False, is_flag=True)
def get_and_save_ais_data(
    chc_table,
    ais_table,
    data_version,
    bounds_table,
    max_dist_to_mesh,
    save_ais_mesh,
    save_soma_mesh,
    soma_table,
    soma_radius,
    n_threads,
    redownload_meshes,
):
    (
        ais_synapse_data,
        oids_failed_ais,
        zero_synapse_oids,
        complete_ais_ids,
    ) = compute_ais_synapse_data(
        chc_table,
        ais_table,
        data_version,
        ais_bound_table=bounds_table,
        max_dist_to_mesh=max_dist_to_mesh,
        save_ais_mesh=save_ais_mesh,
        save_soma_mesh=save_soma_mesh,
        soma_table=soma_table,
        soma_radius=soma_radius,
        n_threads=n_threads,
        redownload_meshes=redownload_meshes,
    )

    aggregated_ais_syn_df = aggregate_ais_dataframes(complete_ais_ids, ais_synapse_data)
    params = {
        "soma_table": soma_table,
        "bounds_table": bounds_table,
        "soma_radius": soma_radius,
        "max_dist_to_mesh": max_dist_to_mesh,
        "chc_table": chc_table,
        "ais_table": ais_table,
        "data_version": data_version,
    }
    save_ais_synapse_data(
        ais_synapse_data,
        aggregated_ais_syn_df,
        oids_failed_ais,
        zero_synapse_oids,
        chc_table,
        ais_table,
        data_version,
        params,
        data_dir,
    )


if __name__ == "__main__":
    get_and_save_ais_data()
