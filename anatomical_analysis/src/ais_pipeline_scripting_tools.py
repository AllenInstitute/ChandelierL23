import os
import re
import pandas as pd
import numpy as np
from itertools import chain
from meshparty import trimesh_io
from scipy import spatial, sparse
from meshparty import mesh_filters, trimesh_repair, skeletonize
import mesh_fix_utils as mesh_repair

import warnings

warnings.simplefilter("ignore")

DEFAULT_VOXEL_RESOLUTION = [3.58, 3.58, 40]
DEFAULT_VOXEL_SCALING = [0.895, 0.895, 1]
PRE_ID_COL = "pre_pt_root_id"
POST_ID_COL = "post_pt_root_id"
SYN_SIZE_COL = "size"


def clean_mesh(
    mesh,
    ais_points,
    soma_point=None,
    soma_radius=16000,
    line_dist_th=15000,
    comp_size_th=500,
):
    is_close_line = mesh_filters.filter_close_to_line(
        mesh, ais_points, line_dist_th=line_dist_th, sphere_ends=True
    )
    if soma_point is not None:
        is_close_center = mesh_filters.filter_spatial_distance_from_points(
            mesh, soma_point.reshape(1, 3), soma_radius
        )
    else:
        is_close_center = np.full(len(is_close_line), False)
    is_close = is_close_line | is_close_center
    cyl_mesh = mesh.apply_mask(is_close)

    is_dust = mesh_filters.filter_components_by_size(cyl_mesh, min_size=comp_size_th)
    meshf = cyl_mesh.apply_mask(is_dust)
    return meshf


def find_ais_indices(mesh, ais_points):
    _, minds_ais = mesh.kdtree.query(ais_points)
    if len(minds_ais) != 2:
        return None
    else:
        return minds_ais


def needs_repair(mesh, minds_ais):
    n, lbls = sparse.csgraph.connected_components(mesh.csgraph)
    return lbls[minds_ais[0]] != lbls[minds_ais[1]]


def compute_soma(mesh, ctr_pt, soma_radius, mesh_ais, save_mesh=False, filename=None):
    if ctr_pt is None:
        print("No center point found...")
        mesh_soma = None
    else:
        d_from_ctr = np.linalg.norm(mesh.vertices - ctr_pt, axis=1)
        is_close = d_from_ctr < soma_radius
        is_close_not_ais = is_close & mesh.filter_unmasked_boolean(~mesh_ais.node_mask)
        if any(is_close_not_ais):
            mesh_soma = mesh.apply_mask(is_close_not_ais)
            not_small = mesh_filters.filter_components_by_size(
                mesh_soma, min_size=20000
            )
            mesh_soma = mesh_soma.apply_mask(not_small)
            if save_mesh:
                mesh_soma.write_to_file(filename)
        else:
            print("No soma vertices found...")
            mesh_soma = None
    return mesh_soma


def clean_and_repair_ais_link_edges(
    mesh,
    ais_points,
    seg_id,
    dataset_name,
    soma_point=None,
    soma_radius=16000,
    line_dist_th=15000,
    comp_size_th=1000,
    mesh_distance_upper_bound=250,
    do_save_on_repair=False,
    filename=None,
):
    if len(mesh.link_edges) == 0:
        mesh.add_link_edges(int(seg_id), dataset_name)

    mesh_clean = clean_mesh(
        mesh,
        ais_points,
        soma_point=soma_point,
        soma_radius=soma_radius,
        line_dist_th=line_dist_th,
        comp_size_th=comp_size_th,
    )
    minds_ais = find_ais_indices(mesh_clean, ais_points)

    if needs_repair(mesh_clean, minds_ais):
        print("No mesh repair solution found")
        return None, None

    if do_save_on_repair:
        trimesh_io.write_mesh_h5(
            filename,
            mesh_bandaged.vertices,
            mesh_bandaged.faces,
            node_mask=mesh_bandaged.node_mask,
            link_edges=mesh_bandaged.link_edges,
            overwrite=True,
        )

    return mesh_clean, minds_ais


def clean_and_repair_ais(
    mesh,
    ais_points,
    soma_point=None,
    soma_radius=16000,
    line_dist_th=15000,
    comp_size_th=1000,
    mesh_distance_upper_bound=250,
    do_save_on_repair=False,
    filename=None,
):
    # Assumes that the mesh is the full original mesh.
    mesh_clean = clean_mesh(
        mesh,
        ais_points,
        soma_point=soma_point,
        soma_radius=soma_radius,
        line_dist_th=line_dist_th,
        comp_size_th=comp_size_th,
    )

    minds_ais = find_ais_indices(mesh_clean, ais_points)
    if minds_ais is None:
        print("One or both AIS points were not found")
        return None, None

    if needs_repair(mesh_clean, minds_ais):
        link_edges = mesh_repair.bandage_mesh(
            mesh_clean, minds_ais, mesh_distance_upper_bound=mesh_distance_upper_bound
        )
        if link_edges is None:
            print("No mesh repair solution found")
            return None, None

        if ~np.all(mesh.node_mask):
            link_edges = mesh.filter_unmasked_indices(link_edges)
        mesh.link_edges = np.vstack([mesh.link_edges, link_edges])

        if do_save_on_repair:
            mesh.write_to_file(filename)

        # All we did is add link_edges, so we should be able to reuse the node mask
        mesh_clean = mesh.apply_mask(mesh_clean.node_mask)

    return mesh_clean, minds_ais


def compute_ais(mesh, minds_ais, d_pad=1000):

    if len(minds_ais) != 2:
        print("Two AIS points are needed")
        return None

    d_ais_pts_to_all = sparse.csgraph.dijkstra(
        mesh.csgraph,
        indices=minds_ais,
        unweighted=False,
    )
    dmax = d_ais_pts_to_all[0, minds_ais[1]] + d_pad

    if np.isinf(dmax):
        print("Top and bottom AIS points are not in the same mesh component")
        return None

    is_ais = np.sum(np.square(d_ais_pts_to_all), axis=0) < dmax ** 2

    # Need this in case a self-contact occurs in a nearby branch.
    mesh_ais = mesh.apply_mask(is_ais)
    is_mesh_ais_lcc = mesh_filters.filter_largest_component(mesh_ais)

    return mesh_ais.map_boolean_to_unmasked(is_mesh_ais_lcc)


def skeletonize_ais(ais_mesh, invalidation_d=10000):
    ais_sk = skeletonize.skeletonize_mesh(
        ais_mesh, invalidation_d=invalidation_d, verbose=False
    )
    try:
        potential_tips = ais_sk.end_points_flat
        top_pt_ind = potential_tips[
            np.argmin(ais_sk.vertices[potential_tips, 1])
        ]  # Pick lowest y-value tip
        ais_sk.reroot(top_pt_ind)
        return ais_sk
    except:
        return None


def distance_along_skeleton(ais_sk, syn_df):
    syn_sk_inds = ais_sk.mesh_to_skel_map[syn_df["post_pt_mesh_ind"]]

    lowest_endpoint = ais_sk.end_points_flat[
        np.argmax(ais_sk.vertices[ais_sk.end_points_flat, 1])
    ]
    core_path = ais_sk.path_to_root(lowest_endpoint)
    in_core = np.isin(syn_sk_inds, core_path)

    sk_distance = np.zeros(
        len(
            syn_sk_inds,
        )
    )

    syn_along_core = syn_sk_inds[in_core]
    sk_distance[in_core] = ais_sk.distance_to_root[syn_along_core]

    effective_d = []
    paths_to_root_non = [
        np.array(ais_sk.path_to_root(syn_pt)) for syn_pt in syn_sk_inds[~in_core]
    ]
    for ptr in paths_to_root_non:
        path_in_core = np.isin(ptr, core_path)
        effective_d.append(np.max(ais_sk.distance_to_root[ptr[path_in_core]]))

    sk_distance[~in_core] = effective_d
    return sk_distance, ais_sk.path_length([core_path])


def attach_synapses_to_mesh(mesh, syns_df, d_th=np.inf, return_unmasked=True):
    syns_pts = scale_voxel_to_euclidean(syns_df["ctr_pt_position"])
    ds, mesh_inds = mesh.kdtree.query(syns_pts, distance_upper_bound=d_th)
    mesh_inds_out = np.full(mesh_inds.shape, -1)
    mesh_inds_out[~np.isinf(ds)] = mesh.map_indices_to_unmasked(
        mesh_inds[~np.isinf(ds)]
    )
    return mesh_inds_out


def compute_per_ais_properties(aggregated_ais_syn_df):
    aggregated_ais_syn_df["size_flat"] = aggregated_ais_syn_df["size"].apply(
        lambda x: [x for x in chain.from_iterable(x)]
    )
    conn_sizes = [x for x in chain.from_iterable(aggregated_ais_syn_df["size_flat"])]
    size_th_percentiles = [25, 50, 75]
    size_ths = {th: np.percentile(conn_sizes, q=th) for th in size_th_percentiles}

    aggregated_ais_syn_df["num_cells"] = aggregated_ais_syn_df["syns"].map(len)

    aggregated_ais_syn_df["syn_net"] = aggregated_ais_syn_df["syns"].map(sum).fillna(0)
    aggregated_ais_syn_df["syn_mean"] = aggregated_ais_syn_df["syns"].map(np.mean)
    aggregated_ais_syn_df["syn_median"] = aggregated_ais_syn_df["syns"].map(np.median)
    aggregated_ais_syn_df["syn_var"] = aggregated_ais_syn_df["syns"].map(np.var)
    aggregated_ais_syn_df["syn_max"] = aggregated_ais_syn_df["syns"].map(
        lambda x: np.max(x) if len(x) > 0 else 0
    )

    aggregated_ais_syn_df["size_net"] = (
        aggregated_ais_syn_df["size_flat"].map(sum).fillna(0)
    )
    aggregated_ais_syn_df["size_mean"] = aggregated_ais_syn_df["size_flat"].map(np.mean)
    aggregated_ais_syn_df["size_median"] = aggregated_ais_syn_df["size_flat"].map(
        np.median
    )
    aggregated_ais_syn_df["size_var"] = aggregated_ais_syn_df["size_flat"].map(np.var)
    aggregated_ais_syn_df["size_max"] = aggregated_ais_syn_df["size_flat"].map(
        lambda x: np.max(x) if len(x) > 0 else 0
    )

    return aggregated_ais_syn_df


def edge_list_from_df(synapse_df, threshold=0, return_size=False):
    """
    Generate a weighted edge list between root ids given a synapse query reponse
    """
    groups = synapse_df[[PRE_ID_COL, POST_ID_COL, SYN_SIZE_COL]].groupby(
        [PRE_ID_COL, POST_ID_COL]
    )
    edge_list = groups.agg(list)
    edge_list["syns"] = edge_list.applymap(len)
    if not return_size:
        edge_list.drop(columns=SYN_SIZE_COL)
    return edge_list[edge_list.syns >= threshold].reset_index()


def synapse_compartment_distribution(syn_df):
    if "label" not in syn_df.columns:
        raise AttributeError("The synapse dataframe needs a 'label' column")
    syn_by_label_long = syn_df.groupby((PRE_ID_COL, "label")).count().id.reset_index()
    syn_by_label = syn_by_label_long.pivot(
        index=PRE_ID_COL, columns="label", values="id"
    ).fillna(0)
    return syn_by_label


def total_input_synapses(
    post_id_list, synapse_df, pre_id_list=None, on_compartments=None
):
    return _total_synapses(
        post_id_list,
        synapse_df,
        connection_type="post",
        other_id_list=pre_id_list,
        on_compartments=on_compartments,
    )


def total_output_synapses(
    pre_id_list, synapse_df, post_id_list=None, on_compartments=None
):
    return _total_synapses(
        pre_id_list,
        synapse_df,
        connection_type="pre",
        other_id_list=post_id_list,
        on_compartments=on_compartments,
    )


def _total_synapses(
    id_list, synapse_df, connection_type=None, other_id_list=None, on_compartments=None
):
    if connection_type not in ["pre", "post"]:
        raise ValueError("Type must be pre or post")

    if other_id_list is not None:
        synapse_df = synapse_df[synapse_df[other_col].isin(other_id_list)]
    if on_compartments is not None:
        synapse_df = synapse_df[synapse_df["label"].isin(on_compartments)]

    if connection_type == "pre":
        grouped_list = output_distribution(id_list, synapse_df)
        col_name = "num_outputs"
    else:
        grouped_list = input_distribution(id_list, synapse_df)
        col_name = "num_inputs"
    grouped_list[col_name] = grouped_list["syns"].map(sum)
    return grouped_list


def aggregate_postsynaptic(
    id_list, synapse_df, threshold=0, on_compartments=None, return_size=False
):
    return _aggregate_synapses(
        id_list, synapse_df, "post", threshold, on_compartments, return_partners=True
    )


def aggregate_presynaptic(
    id_list, synapse_df, threshold=0, on_compartments=None, return_size=False
):
    return _aggregate_synapses(
        id_list, synapse_df, "pre", threshold, on_compartments, return_partners=True
    )


def _aggregate_synapses(
    id_list,
    synapse_df,
    connection_type,
    threshold=0,
    on_compartments=None,
    return_partners=False,
    return_size=False,
):
    if connection_type not in ["pre", "post"]:
        raise ValueError("Type must be pre or post")
    if connection_type == "pre":
        focus_col = PRE_ID_COL
    else:
        focus_col = POST_ID_COL

    if on_compartments is not None:
        synapse_df = synapse_df[synapse_df["label"].isin(on_compartments)]

    edge_list = edge_list_from_df(synapse_df, threshold=threshold)
    grouped_list = edge_list.groupby(focus_col).agg(list)

    # Concatenate with a null dataframe for ids not in the synapse list
    zero_indices = set(id_list).difference(grouped_list.index)
    if len(zero_indices) > 0:
        zero_df = pd.DataFrame(
            columns=grouped_list.columns, index=pd.Index(zero_indices, name=focus_col)
        )
        for col in zero_df.columns:
            zero_df[col][zero_df[col].isnull()] = zero_df[col][
                zero_df[col].isnull()
            ].apply(lambda x: [])
        grouped_list = pd.concat((grouped_list, zero_df))

    if return_partners:
        return grouped_list.reset_index()
    else:
        return grouped_list["syns"].reset_index()


def synapse_spatial_statistics(syn_pts, mesh, kdt=None):
    if kdt is None:
        vertex_kdt = spatial.cKDTree(mesh.vertices)
    else:
        vertex_kdt = kdt
    ds, rows = vertex_kdt.query(syn_pts)

    d_syn_to_all = sparse.csgraph.dijkstra(
        mesh.csgraph, indices=rows, unweighted=False, directed=False
    )
    d_syn = d_syn_to_all[:, rows]

    d_max = max(d_syn.flatten())

    # Replace this with something related to distance to soma points when possible.
    syn_highest = np.argmin(syn_pts[:, 1])  # Get synapse with highest y value

    d_from_top = d_syn[syn_highest, :]

    syn_order = np.argsort(d_from_top)
    try:
        d_intersyn = [
            d_syn[syn_order[ii], syn_order[ii + 1]]
            for ii in np.arange(0, len(syn_order) - 1)
        ]
    except:
        d_intersyn = np.array([])
    return dict(d_max=d_max, d_from_top=d_from_top, d_intersyn=d_intersyn)


def _ids_in_mesh_dir(mesh_dir):
    fns = os.listdir(mesh_dir)
    id_list = []
    for fn in fns:
        qry = re.match("^(\d*).h5", fn)
        if qry is not None:
            id_list.append(int(qry.groups()[0]))
    return id_list


def make_id_map(id_table_name, dl_dicts):
    """
    dl dicts are {'suffix':dl}
    """
    for suf, dl in dl_dict.items():
        suffix_1 = suf
        dl_1 = dl

    suffix_1 = list(dl_1_dict.keys())[0]
    ids_1 = dl36.query_cell_ids(id_table_name)
    ids_2 = dl_new.query_cell_ids(id_table_name)


def ais_synapses_from_bounds(mesh, syn_df, ais_pts, d_pad=300, mesh_size_th=1000):
    _, component = sparse.csgraph.connected_components(mesh.csgraph)
    component_ids, cnt = np.unique(component, return_counts=True)
    keep_components = component_ids[cnt > mesh_size_th]
    keep_slice = np.isin(component, keep_components)
    mesh_vertices = mesh.vertices[keep_slice]
    mesh_graph = mesh.csgraph[:, keep_slice][keep_slice]

    kdt = spatial.cKDTree(mesh_vertices)
    pt_ds, ais_bound_mesh_inds = kdt.query(ais_pts)

    if len(ais_bound_mesh_inds) < 2:
        print("One or both AIS points were not found")
        return None, None
    print("Distance from ais_pts to {}".format(pt_ds))

    syn_pts = np.vstack(syn_df.ctr_pt_position.values) * np.array([4, 4, 40])
    syn_pt_ds, syn_mesh_inds = kdt.query(syn_pts)

    d_ais_to_all = sparse.csgraph.dijkstra(
        mesh_graph, indices=ais_bound_mesh_inds, unweighted=False, directed=False
    )
    dmax = d_ais_to_all[0, ais_bound_mesh_inds[1]] + d_pad
    if np.isinf(dmax):
        print("Top and bottom of AIS are not in same mesh component.")
        return None, None
    d_ais_to_syn = d_ais_to_all[:, syn_mesh_inds]
    syn_df["is_ais_input"] = np.logical_and(
        d_ais_to_syn[0, :] < dmax, d_ais_to_syn[1, :] < dmax
    )

    return syn_df, pt_ds


def filter_mesh_graph_components(mesh, node_threshold=1000):
    _, component = sparse.csgraph.connected_components(mesh.csgraph)
    component_ids, cnt = np.unique(component, return_counts=True)
    keep_components = component_ids[cnt > node_threshold]
    keep_slice = np.isin(component, keep_components)
    mesh_vertices = mesh.vertices[keep_slice]
    mesh_graph = mesh.csgraph[:, keep_slice][keep_slice]
    return mesh_vertices, mesh_graph


def csgraph_append_new_edges(new_edges, mesh, csgraph=None):
    if csgraph == None:
        csgraph = mesh.csgraph

    weights = np.linalg.norm(
        mesh.vertices[new_edges[:, 0]] - mesh.vertices[new_edges[:, 1]], axis=1
    )

    edges = np.concatenate([new_edges.T, new_edges.T[[1, 0]]], axis=1)
    weights = np.concatenate([weights, weights]).astype(dtype=np.float32)

    new_csgraph = sparse.csr_matrix(
        (weights, edges),
        shape=[
            len(mesh.vertices),
        ]
        * 2,
        dtype=np.float32,
    )

    return csgraph + new_csgraph


def repair_big_flat_faces(
    mesh,
    vertices_component_a,
    vertices_component_b,
    d_max=300,
    orientation_th=0.99,
    orientation_vec=np.array([0, 0, 1]),
    area_threshold_prc=95,
):
    # Make sure faces are flat enough
    z_prod = np.abs(np.dot(mesh.face_normals, orientation_vec))
    flat_faces = z_prod > orientation_th

    # Make sure triangles are big enough
    area_threshold = np.percentile(mesh.area_faces, area_threshold_prc)
    big_faces = mesh.area_faces > area_threshold

    big_flat_vertices = np.unique(
        mesh.faces[np.logical_and(big_faces, flat_faces), :].flatten()
    )
    is_flat_vertices = np.full(len(mesh.vertices), False)
    is_flat_vertices[big_flat_vertices] = True

    vertices_component_a = np.logical_and(vertices_component_a, is_flat_vertices)
    vertices_component_b = np.logical_and(vertices_component_b, is_flat_vertices)

    new_edges = _repair_mesh_graph_edges(
        mesh, vertices_component_a, vertices_component_b, d_max=d_max
    )
    return new_edges


def _repair_mesh_graph_edges(
    mesh,
    vertices_component_a,
    vertices_component_b,
    d_max=300,
):
    vca_map = vertices_component_a.nonzero()[0]
    vcb_map = vertices_component_b.nonzero()[0]

    kdt_a = spatial.cKDTree(mesh.vertices[vertices_component_a, :])
    ds, a_pts = kdt_a.query(
        mesh.vertices[vertices_component_b, :], distance_upper_bound=d_max
    )

    new_edge_map = []
    for b_ind in (~np.isinf(ds)).nonzero()[0]:
        new_edge_map.append([vcb_map[b_ind], vca_map[a_pts[b_ind]]])
    return new_edge_map


def append_spatial_data(ais_syn_df, ais_mesh, ais_pts=None, distance_type="mesh"):

    ais_syn_df = ais_syn_df.reset_index().drop(columns=["index"])

    if ais_pts is None:
        ais_top_mesh_pt = np.argmin(ais_mesh.vertices[:, 1])
    else:
        top_ais_pt_ind = np.argmin(ais_pts[:, 1])
        _, ais_top_mesh_pt = ais_mesh.kdtree.query(np.array([ais_pts[top_ais_pt_ind]]))

    if distance_type == "mesh":
        d_syns = distance_on_mesh(
            ais_mesh,
            ais_mesh.filter_unmasked_indices(ais_syn_df["post_pt_mesh_ind"].values),
        )
    elif distance_type == "euclidean":
        d_syns = spatial.dist.pdist(syn_points)

    if len(ais_syn_df) != 0:
        syn_order = point_order_along_axis(ais_syn_df, d_syns)
    else:
        syn_order = []
    ais_syn_df["d_top"] = dist_from_top(
        ais_mesh,
        ais_mesh.filter_unmasked_indices(ais_syn_df["post_pt_mesh_ind"].values),
        ais_top_mesh_pt,
    )
    ais_syn_df["d_first"] = dist_from_first(d_syns, syn_order)
    ais_syn_df["d_previous"] = interpoint_interval(d_syns, syn_order)
    ais_syn_df["d_previous_same_cell"] = dist_points_same_cell(
        ais_syn_df, d_syns, syn_order
    )
    ais_syn_df["d_closest"] = dist_points_closest(ais_syn_df, d_syns)
    ais_syn_df["d_closest_diff_cell"] = dist_points_closest_diff_cell(
        ais_syn_df, d_syns
    )

    return ais_syn_df


def point_order_along_axis(ais_syn_df, dmat, axis=1):
    pts = np.vstack(ais_syn_df["ctr_pt_position"].values)
    top_ind = np.argmin(pts[:, axis])
    return np.argsort(dmat[top_ind, :])


def dist_from_top(ais_mesh, syn_inds, top_ais_ind):
    ds = sparse.csgraph.dijkstra(ais_mesh.csgraph, indices=[top_ais_ind])
    return ds.flatten()[syn_inds]


def dist_from_first(dmat, order):
    return dmat[order[0], :]


def dist_from_previous(syn_df, dmat, order):
    syn_df[col_name] = interpoint_interval(dmat, order)


def interpoint_interval(dmat, order):
    d_from_previous = np.array(
        [0] + [dmat[order[ii], ind] for ii, ind in enumerate(order[1:])]
    )
    d_from_previous = d_from_previous[np.argsort(order)]
    return d_from_previous


def dist_points_same_cell(syn_df, dmat, order):
    anno_ids = []
    d_previous_same_cell = []
    if "d_previous_same_cell" in syn_df.columns:
        syn_df = syn_df.drop(columns=["d_previous_same_cell"])
    for preoid in np.unique(syn_df.pre_pt_root_id):
        row = syn_df.pre_pt_root_id == preoid
        order_row = order[np.isin(order, np.flatnonzero(row))]
        d_previous_same_cell.append(interpoint_interval(dmat, order_row))
        anno_ids.append(syn_df[row].id.values)
    d_previous_same_cell = [x for x in chain.from_iterable(d_previous_same_cell)]
    anno_ids = [x for x in chain.from_iterable(anno_ids)]
    df_dfpsc = pd.DataFrame(
        {"id": anno_ids, "d_previous_same_cell": d_previous_same_cell}
    )
    return syn_df.merge(df_dfpsc, on="id", how="left")[["d_previous_same_cell"]]


def dist_points_closest(syn_df, dmat):
    dmat_c = dmat.copy()
    np.fill_diagonal(dmat_c, np.inf)
    closest_dist = np.min(dmat_c, axis=1)
    return closest_dist


def dist_points_closest_diff_cell(syn_df, dmat):
    closest_dist_other_cells = []
    for ind, row in syn_df.reset_index().iterrows():
        is_other = (syn_df.pre_pt_root_id != row.pre_pt_root_id).values
        if any(is_other):
            closest_dist_other_cells.append(np.min(dmat[ind, is_other]))
        else:
            closest_dist_other_cells.append(np.nan)
    return closest_dist_other_cells


def synapses_in_edge(syn_df):
    if "syn_per_edge" in syn_df.columns:
        syn_df = syn_df.drop(columns=["syn_per_edge"])
    right_df = (
        syn_df[["pre_pt_root_id", "id"]].groupby("pre_pt_root_id").count().reset_index()
    )
    right_df = right_df.rename(columns={"id": "syn_per_edge"})
    return syn_df.merge(right_df, how="left", on="pre_pt_root_id")["syn_per_edge"]


def clean_df(df):
    ignore_columns = [
        "index",
        "is_ais_input",
        "label",
        "synapse_id",
        "valid",
        "pre_pt_supervoxel_id",
        "post_pt_supervoxel_id",
    ]
    for col in ignore_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def csr_zero_rows(csr, rows_to_zero):
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)


def slice_local_mesh_along_plane(
    vertex_index, slice_plane_normal, d_max, mesh, rotation_matrix=None
):
    """
    Given a vertex index, take a slice d_max away from that vertex of the mesh along a specific plane (specified by its normal vector)
    and within the same connected component after slicing. Returns a boolean list for mesh vertices.
    """
    if rotation_matrix is None:
        vertices = mesh.vertices
    else:
        vertices = np.dot(rotation_matrix, mesh.vertices.T).T
    ctr_pt = vertices[vertex_index]
    slice_plane_normal = slice_plane_normal / np.linalg.norm(slice_plane_normal)
    mv_centered = vertices - ctr_pt
    mv_dist = np.abs(np.linalg.multi_dot([mv_centered, slice_plane_normal]))
    mv_far = mv_dist > d_max

    mg_split = mesh.csgraph.copy()
    csr_zero_rows(mg_split, mv_far)
    n, lbls = sparse.csgraph.connected_components(mg_split)

    keep_inds = lbls == lbls[vertex_index]
    return keep_inds


def point_orientation_from_zx_slice(vertex_inds, mesh, d_max, rotation_matrix=None):
    slice_normal = np.array([0, 1, 0])
    point_orientation = []

    if rotation_matrix is None:
        vertices = mesh.vertices
    else:
        vertices = np.dot(rotation_matrix, mesh.vertices.T).T

    for ind in vertex_inds:
        slice_inds = slice_local_mesh_along_plane(
            ind, slice_normal, d_max, mesh, rotation_matrix
        )
        zx_pnts = vertices[slice_inds][:, [2, 0]]
        slice_hull = spatial.ConvexHull(zx_pnts)
        slice_hull_pts = zx_pnts[slice_hull.vertices]
        slice_centroid = np.mean(slice_hull_pts, axis=0)

        syn_pt = vertices[ind]
        dir_vec = syn_pt[[2, 0]] - slice_centroid
        point_orientation.append(np.arctan2(dir_vec[1], dir_vec[0]))
    return point_orientation


def non_chc_location_fraction(rel_data):
    top_chc = np.argmin(rel_data[rel_data["is_chandelier"] == True]["d_top"])
    d_top = rel_data[rel_data["is_chandelier"] == True].loc[top_chc]["d_top"]
    bot_chc = np.argmax(rel_data[rel_data["is_chandelier"] == True]["d_top"])
    d_bot = rel_data[rel_data["is_chandelier"] == True].loc[bot_chc]["d_top"]

    d_non_orig = rel_data[rel_data["is_chandelier"] == False]["d_top"]

    before_inds = d_non_orig < d_top
    mid_inds = (d_non_orig >= d_top) & (d_non_orig <= d_bot)
    after_inds = d_non_orig > d_bot
    total_num = np.array([sum(before_inds), sum(mid_inds), sum(after_inds)])
    return total_num


def scale_voxel_to_euclidean(pts, voxel_res=DEFAULT_VOXEL_RESOLUTION):
    """
    Scale points from voxel space to euclidean space.
    """
    return np.vstack(pts) * voxel_res


def distance_on_mesh(mesh, to_indices, from_indices=None, limit=np.inf):
    """
    Compute a distance matrix along a mesh graph.
    """
    if from_indices is None:
        from_indices = to_indices
    if from_indices == "all":
        dmat = sparse.csgraph.dijkstra(mesh.csgraph, indices=to_indices, limit=limit)
        d_pts = dmat.T
    else:
        dmat = sparse.csgraph.dijkstra(mesh.csgraph, indices=from_indices, limit=limit)
        d_pts = dmat[:, to_indices]
    return d_pts
