from argparse import ArgumentParser
from geometry_processing import Manifold
from matplotlib.cm import jet
from matplotlib.colors import Normalize
from matplotlib.pyplot import show, subplots
from potpourri3d import read_mesh
from torch import tensor


parser = ArgumentParser()
parser.add_argument('--mesh', default='sphere.obj')
args = parser.parse_args()

vertices, faces = read_mesh('meshes/' + args.mesh)
vertices = tensor(vertices)
faces = tensor(faces)

m = Manifold(faces, dtype=vertices.dtype)
fs = vertices
us = m.embedding_to_halfedge_vectors(fs)
L = m.angles_to_laplacian(m.halfedge_vectors_to_angles(us))
vertex_As = m.areas_to_vertex_areas(m.halfedge_vectors_to_areas(us))

num_rows = 6
num_cols = 8
num_eigs = num_rows * num_cols
eigvals, eigvecs = m.laplacian_and_vertex_areas_to_eigs(L, vertex_As, num_eigs=num_eigs + 1, num_iters=200, verbose=True, eps=1e-8)
eigvals = eigvals[1:]
eigvecs = eigvecs[:, 1:]
print(eigvals)

_, axs = subplots(num_rows, num_cols, figsize=(1.5 * num_cols, 1.5 * num_rows), tight_layout=True, subplot_kw=dict(projection='3d'))
for ax, eigvec in zip(axs.flatten(), eigvecs.T):
    surf = ax.plot_trisurf(*fs.T, triangles=faces)
    surf.set_fc(jet(Normalize(vmin=eigvec.min(), vmax=eigvec.max())(eigvec[faces].mean(dim=-1))))
    ax.view_init(azim=45)
    ax.set_box_aspect((1, 1, 1))
    ax.axis('off')

show()
