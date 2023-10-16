from argparse import ArgumentParser
from geometry_processing import Manifold, PuncturedTopologicalSphere
from matplotlib.pyplot import figure, show
from potpourri3d import read_mesh
from torch import tensor


parser = ArgumentParser()
parser.add_argument('--mesh', default='cube.obj')
args = parser.parse_args()

vertices, faces = read_mesh('meshes/' + args.mesh)
vertices = tensor(vertices)
faces = tensor(faces)

sphere = Manifold(faces, dtype=vertices.dtype)
disk = PuncturedTopologicalSphere(sphere)
fs = vertices[1:]
ls = disk.halfedge_vectors_to_metric(disk.embedding_to_halfedge_vectors(fs))
_, flat_ls = disk.metric_to_flat_metric(ls, num_iters=50, verbose=True)
flat_fs = disk.metric_to_spectral_conformal_parametrization_fiedler(flat_ls, num_iters=20, verbose=True)

fig = figure(figsize=(8, 4), tight_layout=True)

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_trisurf(*vertices.T, triangles=faces)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

ax = fig.add_subplot(1, 2, 2)
ax.triplot(*flat_fs.T, triangles=disk.faces, linewidth=0.1)
ax.axis('equal')

show()
