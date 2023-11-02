from argparse import ArgumentParser
from geometry_processing import Manifold
from geometry_processing.utils import embedding_to_spherical_parametrization
from matplotlib.pyplot import show, subplots
from potpourri3d import read_mesh
from torch import exp, tensor


parser = ArgumentParser()
parser.add_argument('--mesh', default='cube.obj')
args = parser.parse_args()

vertices, faces = read_mesh('meshes/' + args.mesh)
vertices = tensor(vertices)
faces = tensor(faces)

sphere = Manifold(faces, dtype=vertices.dtype)
fs = vertices
sphere_fs, log_factors = embedding_to_spherical_parametrization(sphere, fs, flatten_num_iters=50, layout_num_iters=20, verbose=True)

init_metric = sphere.embedding_to_metric(fs)
final_metric = sphere.embedding_to_metric(sphere_fs)
discrete_conformal_err = final_metric - exp(sphere.signed_vertices_to_halfedges.abs() @ log_factors / 2) * init_metric
print(discrete_conformal_err.abs().max().item())

_, axs = subplots(1, 2, figsize=(8, 4), tight_layout=True, subplot_kw=dict(projection='3d'))
for ax, data in zip(axs, [fs, sphere_fs]):
    ax.plot_trisurf(*data.T, triangles=faces, linewidth=0.1, edgecolor='w')
    ax.view_init(azim=45)
    ax.set_box_aspect((1, 1, 1))
    ax.axis('off')

show()
