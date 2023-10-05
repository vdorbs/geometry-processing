from geometry_processing.manifold import Manifold
from torch import arange, cat, exp, eye, log, ones, pi, randn, sort, sparse_coo_tensor, sqrt, stack, Tensor, unique, zeros
from torch.linalg import norm
from torchsparsegradutils.cupy import sparse_solve_c4t
from typing import Tuple


class PuncturedToplogicalSphere(Manifold):
    def __init__(self, sphere: Manifold):
        faces = sphere.faces[(sphere.faces != 0).all(dim=-1)] - 1
        Manifold.__init__(self, faces, dtype=sphere.dtype)

        self.boundary_vertices = unique(sort(sphere.faces[(sphere.faces == 0).any(dim=-1)]).values)[1:] - 1
        self.is_interior_vertex = (arange(self.num_vertices).reshape(-1, 1) != self.boundary_vertices.reshape(1, -1)).all(dim=-1)
        self.num_interior_vertices = self.is_interior_vertex.sum()

        has_boundary_edge = (faces.reshape(-1, 3, 1) == self.boundary_vertices.reshape(1, 1, -1)).any(dim=-1).sum(dim=-1) == 2
        faces_with_boundary_edges = faces[has_boundary_edge]
        assert len(faces_with_boundary_edges) == len(self.boundary_vertices)
        in_boundary_edge = (faces_with_boundary_edges.reshape(-1, 3, 1) == self.boundary_vertices.reshape(1, 1, -1)).any(dim=-1)

        halfedge_offsets = (1 - (1 * in_boundary_edge)) @ arange(3)
        self.boundary_halfedges = arange(len(faces))[has_boundary_edge] + halfedge_offsets

        boundary_pair_idxs = []
        for face, selector in zip(faces_with_boundary_edges, in_boundary_edge):
            if ~selector[1]:
                face = face.flip(0)
            boundary_pair_idxs.append(face[selector])
        self.boundary_pair_idxs = stack(boundary_pair_idxs)

        # Area matrix
        values = 1j * cat([-ones(len(self.boundary_pair_idxs), dtype=self.dtype), ones(len(self.boundary_pair_idxs), dtype=self.dtype)]) / 4
        idxs = cat([self.boundary_pair_idxs.T, self.boundary_pair_idxs.flip(1).T], dim=-1)
        self.A = sparse_coo_tensor(idxs, values, (self.num_vertices, self.num_vertices))

    def metric_to_boundary_mass_matrix(self, ls: Tensor):
        boundary_halfedge_lengths = ls[self.boundary_halfedges]
        template = (ones(2, 2, dtype=self.dtype) + eye(2, dtype=self.dtype)).reshape(-1) / 6
        values = (1. + 0j) * boundary_halfedge_lengths.repeat_interleave(4) * template.repeat(len(self.boundary_halfedges))
        row_idxs = self.boundary_pair_idxs.unsqueeze(-1).repeat_interleave(2, dim=-1).reshape(-1)
        col_idxs = self.boundary_pair_idxs.unsqueeze(-2).repeat_interleave(2, dim=-2).reshape(-1)
        idxs = stack([row_idxs, col_idxs])
        B = sparse_coo_tensor(idxs, values, (self.num_vertices, self.num_vertices))

        return B

    def metric_to_flat_metric(self, ls: Tensor, num_iters: int, step_size: float = 0.5, verbose=False) -> Tuple[Tensor, Tensor]:
        lambdas = 2 * log(ls)
        us = zeros(self.num_vertices, dtype=self.dtype)
        for iteration in range(num_iters):
            new_lambdas = lambdas + self.tail_vertices_to_halfedges @ us + self.tip_vertices_to_halfedges @ us
            new_ls = exp(new_lambdas / 2)
            new_angles = self.metric_to_angles(new_ls, keep_safe=False)
            grad = (2 * pi - self.angles_to_angle_sums(new_angles)) / 2
            hess = -self.angles_to_laplacian(new_angles) / 2

            # Impose 0 Dirichlet boundary conditions
            grad = grad[self.is_interior_vertex]
            selector = (hess.indices().reshape(2, -1, 1) != self.boundary_vertices.reshape(1, 1, -1)).all(dim=-1).all(dim=0)
            hess_idxs = hess.indices()[:, selector]
            hess_idxs -= (hess_idxs.reshape(2, -1, 1) > self.boundary_vertices.reshape(1, 1, -1)).sum(dim=-1)
            hess = sparse_coo_tensor(hess_idxs, hess.values()[selector], (self.num_interior_vertices, self.num_interior_vertices))

            # Partial Newton step
            new_us = us[self.is_interior_vertex] - step_size * sparse_solve_c4t(hess, grad)
            us[self.is_interior_vertex] = new_us

            if verbose:
                print(iteration, norm(grad).item())

        new_lambdas = lambdas + self.tail_vertices_to_halfedges @ us + self.tip_vertices_to_halfedges @ us
        new_ls = exp(new_lambdas / 2)

        return us, new_ls

    def metric_to_spectral_conformal_parametrization_fiedler(self, ls: Tensor, num_iters: int, verbose=False) -> Tensor:
        L = self.angles_to_laplacian(self.metric_to_angles(ls))
        M = (1. + 0j) * self.areas_to_diag_mass_matrix(self.metric_to_areas(ls))
        L_conf = -0.5 * L - self.A
        L_conf_shifted = L_conf + 1e-6 * M

        fs = randn(self.num_vertices, 2, dtype=self.dtype)
        fs = fs[:, 0] + 1j * fs[:, 1]
        for iteration in range(num_iters):
            next_fs = sparse_solve_c4t(L_conf_shifted, M @ fs)
            lin_solve_err = norm(L_conf_shifted @ next_fs - M @ fs)

            next_fs -= (next_fs * M.sum(dim=-1)).sum(dim=-1) / M.sum()
            next_fs /= sqrt((next_fs.conj() * (M @ next_fs)).sum(dim=-1).abs())
            fs = next_fs

            if verbose:
                eigenvalue = (fs.conj() * (L_conf @ fs)).sum().real
                residual = L_conf @ fs - eigenvalue * M @ fs
                print(iteration, lin_solve_err.item(), norm(residual).item(), eigenvalue.item())

        fs = stack([fs.real, fs.imag], dim=-1)
        return fs

    def metric_to_spectral_conformal_parametrization_generalized(self, ls: Tensor) -> Tensor:
        raise NotImplementedError
