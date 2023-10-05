from argparse import ArgumentParser
from geometry_processing import Manifold
from potpourri3d import read_mesh
from torch import pi, tensor


parser = ArgumentParser()
parser.add_argument('--mesh', default='cube.obj')
args = parser.parse_args()

tol = 1e-9

vertices, faces = read_mesh('meshes/' + args.mesh)
vertices = tensor(vertices)
faces = tensor(faces)

m = Manifold(faces, dtype=vertices.dtype)
fs = vertices
us = m.embedding_to_halfedge_vectors(fs)
ls = m.halfedge_vectors_to_metric(us)

assert (us.reshape(-1, 3, 3).sum(dim=-2)).abs().max() < tol
assert m.check_metric(ls)

areas_from_embedding = m.halfedge_vectors_to_areas(us)
areas_from_metric = m.metric_to_areas(ls)
assert (areas_from_embedding - areas_from_metric).abs().max() < tol

angles_from_embedding = m.halfedge_vectors_to_angles(us)
angles_from_metric = m.metric_to_angles(ls)
assert (angles_from_embedding - angles_from_metric).abs().max() < tol

areas = areas_from_embedding
surface_area = areas.sum()
assert (m.areas_to_mass_matrix(areas).sum() - surface_area).abs() < tol
assert (m.areas_to_diag_mass_matrix(areas).sum() - surface_area).abs() < tol

angles = angles_from_embedding
assert (angles.reshape(-1, 3).sum(dim=-1) - pi).abs().max() < tol

L = m.angles_to_laplacian(angles)
assert L.sum(dim=0).to_dense().abs().max() < tol
assert L.sum(dim=1).to_dense().abs().max() < tol
