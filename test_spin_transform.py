from argparse import ArgumentParser
from geometry_processing import QuaternionicManifold
from matplotlib.cm import jet, ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.pyplot import figure, show
from potpourri3d import read_mesh
from torch import diag, eye, ones, sqrt, randn, tensor
from torch.linalg import det, norm, qr, svd
from torch.sparse import spdiags
from torchsparsegradutils.cupy import sparse_solve_c4t


def plot_colored_surface(ax, vertices, faces, vertex_colors):
    face_colors = vertex_colors[faces].mean(dim=-1)
    face_colors = jet(Normalize(vmin=face_colors.min(), vmax=face_colors.max())(face_colors))

    surf = ax.plot_trisurf(*vertices.T, triangles=faces)
    surf.set_fc(face_colors)
    return surf


parser = ArgumentParser()
parser.add_argument('--mesh', default='sphere.obj')
args = parser.parse_args()

vertices, faces = read_mesh('meshes/' + args.mesh)
vertices = tensor(vertices)
faces = tensor(faces)

m = QuaternionicManifold(faces, dtype=vertices.dtype)
fs = vertices

# Randomly sample rhos
us = m.embedding_to_halfedge_vectors(fs)
L = m.angles_to_laplacian(m.halfedge_vectors_to_angles(us))
vertex_areas = m.areas_to_vertex_areas(m.halfedge_vectors_to_areas(us)).unsqueeze(-1)
A = -L + 1e-8 * spdiags(ones(len(L), dtype=m.dtype), tensor(0), shape=L.shape)
num_eigs = 25
eigvecs = randn(len(fs), num_eigs, dtype=m.dtype)
for iteration in range(100):
    new_eigvecs = sparse_solve_c4t(A, vertex_areas * eigvecs)
    lin_solve_err = norm(A @ new_eigvecs - vertex_areas * eigvecs, dim=0).max()
    new_eigvecs = qr(sqrt(vertex_areas) * new_eigvecs).Q / sqrt(vertex_areas)
    ortho_err = (new_eigvecs.T @ (vertex_areas * new_eigvecs) - eye(num_eigs, dtype=m.dtype)).abs().max()
    eigvecs = new_eigvecs

    eigvals = diag(eigvecs.T @ (A @ eigvecs))
    residuals = norm(A @ eigvecs - eigvals.unsqueeze(0) * (vertex_areas * eigvecs), dim=0)
    print(iteration, lin_solve_err.item(), ortho_err.item(), residuals)

coeffs = randn(num_eigs, dtype=vertices.dtype)
rhos = eigvecs @ coeffs
new_fs, lambda_quats, eigenvalue = m.embedding_and_vertex_values_to_spun_embedding(fs, rhos, num_iters=5, verbose=True)

fig = figure(figsize=(8, 4), tight_layout=True)

ax = fig.add_subplot(1, 2, 1, projection='3d')
plot_colored_surface(ax, fs, faces, rhos)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

ax = fig.add_subplot(1, 2, 2, projection='3d')
plot_colored_surface(ax, new_fs, faces, rhos)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

show()

us = m.embedding_to_halfedge_vectors(fs)
Hs = m.halfedge_vectors_to_vertex_mean_curvatures(us) / m.areas_to_vertex_areas(m.halfedge_vectors_to_areas(us))

new_us = m.embedding_to_halfedge_vectors(new_fs)
new_Hs = m.halfedge_vectors_to_vertex_mean_curvatures(new_us) / m.areas_to_vertex_areas(m.halfedge_vectors_to_areas(new_us))

recon_rhos = new_Hs * det(lambda_quats).real - Hs
recon_new_fs, _, _ = m.embedding_and_vertex_values_to_spun_embedding(fs, recon_rhos, num_iters=5, verbose=True)

new_fs -= new_fs.mean(dim=0)
recon_new_fs -= recon_new_fs.mean(dim=0)
area_ratio = m.halfedge_vectors_to_areas(us).sum() / m.halfedge_vectors_to_areas(m.embedding_to_halfedge_vectors(recon_new_fs)).sum()
recon_new_fs *= sqrt(area_ratio)
U, _, V_T = svd(new_fs.T @ recon_new_fs)
rotation = U @ V_T
recon_new_fs = (rotation @ recon_new_fs.T).T

recon_err = (rhos - recon_rhos).abs()
rel_recon_err = recon_err / rhos.abs()
print(recon_err.max().item(), rel_recon_err.max().item())

fig = figure(figsize=(8, 4), tight_layout=True)

ax = fig.add_subplot(1, 2, 1, projection='3d')
plot_colored_surface(ax, fs, faces, recon_err)
fig.colorbar(ScalarMappable(Normalize(vmin=recon_err.min(), vmax=recon_err.max()), cmap='jet'), ax=ax)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

ax = fig.add_subplot(1, 2, 2, projection='3d')
plot_colored_surface(ax, fs, faces, rel_recon_err)
fig.colorbar(ScalarMappable(Normalize(vmin=rel_recon_err.min(), vmax=rel_recon_err.max()), cmap='jet'), ax=ax)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

show()

fig = figure(figsize=(8, 8), tight_layout=True)

ax = fig.add_subplot(2, 2, 1, projection='3d')
plot_colored_surface(ax, fs, faces, rhos)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

ax = fig.add_subplot(2, 2, 2, projection='3d')
plot_colored_surface(ax, new_fs, faces, rhos)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

ax = fig.add_subplot(2, 2, 3, projection='3d')
plot_colored_surface(ax, fs, faces, recon_rhos)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

ax = fig.add_subplot(2, 2, 4, projection='3d')
plot_colored_surface(ax, recon_new_fs, faces, recon_rhos)
ax.view_init(azim=45)
ax.set_box_aspect((1, 1, 1))
ax.axis('off')

show()
