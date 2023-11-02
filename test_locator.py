from geometry_processing import Manifold
from geometry_processing.utils import sphere_embedding_to_locator
from potpourri3d import read_mesh
from torch import float64, randn, tensor
from torch.linalg import norm


vertices, faces = read_mesh('meshes/sphere.obj')
vertices = tensor(vertices)
faces = tensor(faces)

sphere = Manifold(faces, dtype=float64)
fs = vertices
locator = sphere_embedding_to_locator(sphere, fs)

query_fs = randn(10, 20, 3, dtype=float64)
query_fs /= norm(query_fs, dim=-1, keepdims=True)
face_idxs, bary_coords = locator(query_fs)
recon_query_fs = (bary_coords.unsqueeze(-1) * fs[faces[face_idxs]]).sum(dim=-2)
recon_query_fs /= norm(recon_query_fs, dim=-1, keepdims=True)
print(norm(recon_query_fs - query_fs, dim=-1).abs().max().item())
