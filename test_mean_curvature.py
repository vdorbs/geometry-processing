from argparse import ArgumentParser
from geometry_processing import Manifold
from matplotlib.pyplot import figure, show
from potpourri3d import read_mesh
from torch import tensor

parser = ArgumentParser()
parser.add_argument('--mesh', default='cube.obj')
args = parser.parse_args()

vertices, faces = read_mesh('meshes/' + args.mesh)
vertices = tensor(vertices)
faces = tensor(faces)

m = Manifold(faces, dtype=vertices.dtype)
fs = vertices
us = m.embedding_to_halfedge_vectors(fs)
Hs = m.halfedge_vectors_to_vertex_mean_curvatures(us)
As = m.areas_to_vertex_areas(m.halfedge_vectors_to_areas(us))

fig = figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
cbar = ax.scatter(*vertices.T, c=Hs / As, cmap='jet')
fig.colorbar(cbar)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')
show()
