from geometry_processing import Manifold, PuncturedTopologicalSphere
from torch import arange, cat, exp, IntTensor, log, Tensor, tensor, zeros
from torch.linalg import cross, norm
from typing import Tuple


def embedding_to_spherical_parametrization(m: Manifold, fs: Tensor, flatten_num_iters: int, layout_num_iters: int, verbose: bool = False) -> Tuple[Tensor, Tensor]:
    disk = PuncturedTopologicalSphere(m)
    disk_fs = fs[1:]

    # Pre-flattening intrinsic processing
    ls = disk.embedding_to_metric(disk_fs)
    log_factors = disk.embedding_to_boundary_normalization(disk_fs, fs[0])
    ls *= exp(disk.signed_vertices_to_halfedges.abs() @ log_factors / 2)

    # Instrinsic flattening
    flattening_log_factors, flat_ls = disk.metric_to_flat_metric(ls, num_iters=flatten_num_iters, verbose=verbose)
    log_factors += flattening_log_factors

    # Extrinsic planar layout
    flat_fs = disk.metric_to_spectral_conformal_parametrization_fiedler(flat_ls, num_iters=layout_num_iters, verbose=verbose)
    flat_fs_3d = cat([flat_fs, zeros(len(flat_fs), 1, dtype=m.dtype)], dim=-1)
    area_ratio = disk.embedding_to_areas(flat_fs_3d).sum() / disk.metric_to_areas(flat_ls).sum()
    log_factors += log(area_ratio) / 2

    # Stereographic projection onto sphere
    flat_f_squared_norms = norm(flat_fs, dim=-1, keepdims=True) ** 2
    stereo_fs = cat([2 * flat_fs, flat_f_squared_norms - 1], dim=-1) / (flat_f_squared_norms + 1)
    log_factors += log(2 / (flat_f_squared_norms[:, 0] + 1))

    north_pole = tensor([[0., 0., 1.]], dtype=m.dtype)
    sphere_fs = cat([north_pole, stereo_fs])
    sphere_fs = tensor([[1., 1., -1.]]) * sphere_fs

    prev_dists = norm(disk_fs[disk.boundary_vertices] - fs[:1], dim=-1)
    curr_dists = norm(stereo_fs[disk.boundary_vertices] - north_pole, dim=-1)
    polar_log_factors = -log_factors[disk.boundary_vertices] - 2 * log(prev_dists / curr_dists)
    polar_log_factor = polar_log_factors.mean(0, keepdims=True)
    log_factors = cat([polar_log_factor, log_factors])

    return sphere_fs, log_factors


def sphere_embedding_to_locator(m: Manifold, sphere_fs: Tensor):
    hyperplane_normals = cross(m.tail_vertices_to_halfedges @ sphere_fs, m.tip_vertices_to_halfedges @ sphere_fs)
    Ns = m.embedding_to_face_normals(sphere_fs, keep_scale=True)

    def locate(fs: Tensor) -> Tuple[IntTensor, Tensor]:
        halfspace_memberships = fs @ hyperplane_normals.transpose(-2, -1) >= 0.
        halfspace_memberships = halfspace_memberships.unflatten(-1, (m.num_faces, 3))
        triangle_memberships = halfspace_memberships.all(dim=-1)
        assert (triangle_memberships.sum(dim=-1) == 1).all()
        face_idxs = (1 * triangle_memberships) @ arange(m.num_faces)

        containing_fs = sphere_fs[m.faces[face_idxs]]
        containing_Ns = Ns[face_idxs]
        proj_factors = (containing_fs[..., 0, :] * containing_Ns).sum(dim=-1) / (fs * containing_Ns).sum(dim=-1)
        projs = proj_factors.unsqueeze(-1) * fs
        proj_diffs = projs.unsqueeze(-2) - containing_fs
        subareas = norm(cross(proj_diffs, cat([proj_diffs[..., 1:, :], proj_diffs[..., :1, :]], dim=-2)), dim=-1)
        bary_coords = subareas[..., tensor([1, 2, 0])] / norm(containing_Ns, dim=-1, keepdims=True)

        return face_idxs, bary_coords

    return locate
