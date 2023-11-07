from argparse import ArgumentParser
from geometry_processing import Manifold
from geometry_processing.utils import embedding_to_spherical_parametrization
from matplotlib.pyplot import show, subplots
from potpourri3d import read_mesh
from torch import tensor


parser = ArgumentParser()
parser.add_argument('--mesh', default='cube.obj')
args = parser.parse_args()

vertices, faces = read_mesh('meshes/' + args.mesh)
fs = tensor(vertices)
m = Manifold(tensor(faces), dtype=fs.dtype)

sphere_fs_1, _ = embedding_to_spherical_parametrization(m, fs, flatten_num_iters=50, layout_num_iters=20)
sphere_fs_2, _ = embedding_to_spherical_parametrization(m, fs, flatten_num_iters=50, layout_num_iters=20)
R = m.embeddings_to_rotation(sphere_fs_1, sphere_fs_2)

_, axs = subplots(2, 2, figsize=(8, 8), tight_layout=True, subplot_kw=dict(projection='3d'))
for ax, data in zip(axs.flatten(), [sphere_fs_1, sphere_fs_2, (R @ sphere_fs_1.T).T, sphere_fs_2]):
    ax.plot_trisurf(*data.T, triangles=m.faces, edgecolor='w', linewidth=0.1)
    ax.view_init(azim=45)
    ax.set_box_aspect((1, 1, 1))
    ax.axis('off')

show()

R = m.embeddings_to_rotation(sphere_fs_1, fs)

_, axs = subplots(2, 2, figsize=(8, 8), tight_layout=True, subplot_kw=dict(projection='3d'))
for ax, data in zip(axs.flatten(), [fs, sphere_fs_1, fs, (R @ sphere_fs_1.T).T]):
    ax.plot_trisurf(*data.T, triangles=m.faces, edgecolor='w', linewidth=0.1)
    ax.view_init(azim=45)
    ax.set_box_aspect((1, 1, 1))
    ax.axis('off')

show()
