# import numpy as np
from scipy import sparse
import numpy as np
from meshparty import trimesh_vtk, trimesh_io
import vtk
from itertools import combinations
from collections import defaultdict


def mutual_closest_edges(mesh_a, mesh_b, distance_upper_bound=250):
    """
    Find pairs of points between two meshes that are one-another's closest point
    on the other mesh.
    :param mesh_a: Trimesh-like
    :param mesh_b: Trimesh-like
    :param distance_upper_bound:
    """
    a_ds, a_inds = mesh_a.kdtree.query(
        mesh_b.vertices, distance_upper_bound=distance_upper_bound
    )
    b_ds, b_inds = mesh_b.kdtree.query(
        mesh_a.vertices, distance_upper_bound=distance_upper_bound
    )
    mutual_closest = b_inds[a_inds[b_inds[~np.isinf(b_ds)]]] == b_inds[~np.isinf(b_ds)]

    a_closest = a_inds[b_inds[~np.isinf(b_ds)]][mutual_closest]
    b_closest = b_inds[~np.isinf(b_ds)][mutual_closest]
    if len(a_closest) > 0:
        mutual_closest_edges = np.unique(np.vstack((a_closest, b_closest)), axis=1).T
        return mutual_closest_edges[:, 0], mutual_closest_edges[:, 1]
    else:
        return [np.array([]), np.array([])]


def all_pairs_mutual_closest_edges(mesh, distance_upper_bound=250):
    """
    Given a mesh, find all components and the mutual closest edges between them.
    """
    _, lbls = sparse.csgraph.connected_components(mesh.csgraph)
    cids = np.unique(lbls)
    submeshes = {}
    for cid in cids:
        submeshes[cid] = mesh.apply_mask(lbls == cid)

    bandaid_edges = defaultdict(dict)
    for cid_A, cid_B in combinations(cids, 2):
        mesh_A = submeshes[cid_A]
        mesh_B = submeshes[cid_B]
        edge_A, edge_B = mutual_closest_edges(mesh_A, mesh_B, distance_upper_bound)
        bandaid_edges[cid_A][cid_B] = edge_A
        bandaid_edges[cid_B][cid_A] = edge_B
    return submeshes, bandaid_edges, lbls


def mesh_edges_from_submeshes(mesh, submeshes, bandaid_edges):
    """
    Given a mesh and its submeshes and bandaid_edges, get all pairs
    """
    # Assumes submeshes are all proper subsets of mesh
    # and indexing as in the output of all_pairs_mutual_closest_edges
    unmasked_bandaid_edges = []
    for cid_A, cid_B in combinations(submeshes.keys(), 2):
        if len(bandaid_edges[cid_A][cid_B]) > 0:
            unmasked_bandaid_edges.append(
                np.vstack(
                    (
                        submeshes[cid_A].map_indices_to_unmasked(
                            bandaid_edges[cid_A][cid_B]
                        ),
                        submeshes[cid_B].map_indices_to_unmasked(
                            bandaid_edges[cid_B][cid_A]
                        ),
                    )
                ).T
            )
    if len(unmasked_bandaid_edges) == 0:
        return None
    if len(unmasked_bandaid_edges) > 1:
        unmasked_bandaid_edges = np.vstack(unmasked_bandaid_edges)
    else:
        unmasked_bandaid_edges = unmasked_bandaid_edges[0]
    return mesh.filter_unmasked_indices(unmasked_bandaid_edges).astype(int)


def path_from_predecessors(Ps, ind_start):
    path = []
    next_ind = ind_start
    while next_ind != -9999:
        path.append(next_ind)
        next_ind = Ps[next_ind]
    return np.array(path)


def filter_close_to_line(mesh, line_end_pts, line_dist_th, axis=1):
    """
    Given a mesh and a line segment defined by two end points, make a filter
    leaving only those nodes within a certain distance of the line segment in
    a plane defined by a normal axis (e.g. the y axis defines distances in the
    xy plane)

    :param mesh: Trimesh-like mesh with N vertices
    :param line_end_pts: 2x3 numpy array defining the two end points
    :param line_dist_th: numeric, distance threshold
    :param axis: integer 0-2. Defines which axis is normal to the plane in
                 which distances is computed. optional, default 1 (y-axis).
    :returns:  N-length boolean array
    """
    line_pt_ord = np.argsort(line_end_pts[:, axis])
    below_top = mesh.vertices[:, axis] > line_end_pts[line_pt_ord[0], axis]
    above_bot = mesh.vertices[:, axis] < line_end_pts[line_pt_ord[1], axis]
    ds = _dist_from_line(mesh.vertices, line_end_pts, axis)
    is_close = (ds < line_dist_th) & below_top & above_bot
    return is_close


def _dist_from_line(pts, line_end_pts, axis):
    ps = (pts[:, axis] - line_end_pts[0, axis]) / (
        line_end_pts[1, axis] - line_end_pts[0, axis]
    )
    line_pts = (
        np.multiply(ps[:, np.newaxis], line_end_pts[1] - line_end_pts[0])
        + line_end_pts[0]
    )
    ds = np.linalg.norm(pts - line_pts, axis=1)
    return ds


def filter_large_components(mesh, size_thresh=1000):
    """
    Returns a mesh filter without any connected components less than a size threshold

    :param mesh: Trimesh-like mesh with N vertices
    :param size_thresh: Integer, min size of a component to keep. Optional, default=1000.
    :returns: N-length boolean array
    """
    cc, labels = sparse.csgraph.connected_components(mesh.csgraph, directed=False)
    uids, counts = np.unique(labels, return_counts=True)
    good_labels = uids[counts > size_thresh]
    return np.in1d(labels, good_labels)


def filter_two_point_distance(mesh, pts_foci, d_pad, power=1):
    """
    Returns a boolean array of mesh points such that the sum of the distance from a
    point to each of the two foci are less than a constant. The constant is set by
    the distance between the two foci plus a user-specified padding. Optionally, use
    other Minkowski-like metrics (i.e. x^n + y^n < d^n where x and y are the distances
    to the foci.)
    :param mesh: Trimesh-like mesh with N vertices
    :param pts_foci: 2x3 np array with the two foci in 3d space.
    :param d_pad: Extra padding of the threhold distance beyond the distance between foci.
    :returns: N-length boolean array
    """
    _, minds_foci = mesh.kdtree.query(pts_foci)

    if len(minds_foci) != 2:
        print("One or both mesh points were not found")
        return None

    d_foci_to_all = sparse.csgraph.dijkstra(
        mesh.csgraph,
        indices=minds_foci,
        unweighted=False,
    )
    dmax = d_foci_to_all[0, minds_foci[1]] + d_pad

    if np.isinf(dmax):
        print("Top and bottom AIS points are not in the same mesh component")
        return None

    if power != 1:
        is_in_ellipse = np.sum(np.power(d_foci_to_all, power), axis=0) < np.power(
            dmax, power
        )
    else:
        is_in_ellipse = np.sum(d_foci_to_all, axis=0) < dmax

    return is_in_ellipse


def vtk_linked_point_actor(
    vertices_a, inds_a, vertices_b, inds_b, line_width=1, color=(0, 0, 0), opacity=0.2
):
    if len(inds_a) != len(inds_b):
        raise ValueError("Linked points must have the same length")

    link_verts = np.vstack((vertices_a[inds_a], vertices_b[inds_b]))
    link_edges = np.vstack(
        (np.arange(len(inds_a)), len(inds_a) + np.arange(len(inds_b)))
    )
    link_poly = trimesh_vtk.graph_to_vtk(link_verts, link_edges.T)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(link_poly)

    link_actor = vtk.vtkActor()
    link_actor.SetMapper(mapper)
    link_actor.GetProperty().SetLineWidth(line_width)
    link_actor.GetProperty().SetColor(color)
    link_actor.GetProperty().SetOpacity(opacity)
    return link_actor


def bandage_mesh(
    mesh, bandage_inds, mesh_distance_upper_bound=250, potential_edge_reweight=True
):
    """
    Given a mesh and two 'bandage' points in two components,
    finds mesh edges
    """

    if len(bandage_inds) != 2:
        print("Bandage requires two points")
        return None
    bandage_len = np.linalg.norm(
        mesh.vertices[bandage_inds[0]] - mesh.vertices[bandage_inds[1]]
    )

    submeshes, bandaid_edges, lbls = all_pairs_mutual_closest_edges(
        mesh, distance_upper_bound=mesh_distance_upper_bound
    )
    mesh_edges_all = mesh_edges_from_submeshes(mesh, submeshes, bandaid_edges)
    if mesh_edges_all is None:
        print("No potential bandage edges found")
        return None

    # Find shortest path in merged components with overcomplete edges
    mesh_potential = trimesh_io.Mesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        link_edges=mesh_edges_all,
        process=False,
    )

    potential_mesh_graph = mesh_potential.csgraph
    if potential_edge_reweight:
        for edge in mesh_edges_all:
            potential_mesh_graph[edge[0], edge[1]] = (
                bandage_len + potential_mesh_graph[edge[0], edge[1]]
            )

    ds, Ps = sparse.csgraph.dijkstra(
        potential_mesh_graph, indices=[bandage_inds[0]], return_predecessors=True
    )
    if not np.isinf(ds[0][bandage_inds[1]]):
        path = path_from_predecessors(Ps[0], bandage_inds[1])

        # Now we know certain edges to link, find other edges nearby
        sp_edges = mesh_edges_all[np.all(np.isin(mesh_edges_all, path), axis=1)]
        ds = sparse.csgraph.dijkstra(
            mesh.csgraph, indices=sp_edges.ravel(), limit=1000, min_only=True
        )
        close_edge_nodes = np.flatnonzero(~np.isinf(ds))
        mesh_edges = mesh_edges_all[
            np.all(np.isin(mesh_edges_all, close_edge_nodes), axis=1)
        ]
        return mesh.map_indices_to_unmasked(mesh_edges)
    else:
        print("No path found between bandage points")
        return None
