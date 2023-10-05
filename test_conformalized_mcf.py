from argparse import ArgumentParser
from geometry_processing import Manifold
from matplotlib.pyplot import show, subplots
from potpourri3d import read_mesh
from torch import tensor
from torchsparsegradutils.cupy import sparse_solve_c4t


parser = ArgumentParser()
parser.add_argument('--mesh', default='cube.obj')
parser.add_argument('--h', type=float, default=0.1)
parser.add_argument('--num_steps', type=int, default=5)
args = parser.parse_args()

vertices, faces = read_mesh('meshes/' + args.mesh)
vertices = tensor(vertices)
faces = tensor(faces)

m = Manifold(faces, dtype=vertices.dtype)
fs = vertices
us = m.embedding_to_halfedge_vectors(fs)
L = m.angles_to_laplacian(m.halfedge_vectors_to_angles(us))

f_traj = [fs]
for step in range(args.num_steps):
    fs = f_traj[-1]
    M = m.areas_to_mass_matrix(m.halfedge_vectors_to_areas(m.embedding_to_halfedge_vectors(fs)))
    fs_next = sparse_solve_c4t(M - args.h * L, M @ fs)
    f_traj.append(fs_next)

fs_selected = [f_traj[0], f_traj[1], f_traj[-1]]
fig, axs = subplots(1, 3, figsize=(12, 4), tight_layout=True, subplot_kw=dict(projection='3d'))
for ax, fs in zip(axs, fs_selected):
    ax.plot_trisurf(*fs.T, triangles=faces, alpha=0.8)
    ax.view_init(azim=45)
    ax.set_box_aspect((1, 1, 1))
    ax.axis('off')

show()
