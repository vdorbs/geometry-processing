from geometry_processing.manifold import Manifold
from torch import arange, cat, diag, eye, randn, sparse_coo_tensor, sqrt, stack, tan, Tensor, tensor, zeros_like
from torch.linalg import norm, qr
from torch.sparse import spdiags
from torchsparsegradutils.cupy import sparse_solve_c4t
from typing import Tuple


class QuaternionicManifold(Manifold):
    def check_quat(self, quat: Tensor, tol: float = 1e-12) -> bool:
        return (quat[..., 0, 0] - quat[..., 1, 1].conj()).abs().max() < tol and (-quat[..., 0, 1] - quat[..., 1, 0].conj()) < tol

    def real(self, quat: Tensor) -> Tensor:
        return quat[..., 0, 0].real

    def imag(self, quat: Tensor) -> Tensor:
        alpha, beta = self.quat_to_complex_pair(quat)

        return stack([beta.real, beta.imag, alpha.imag], dim=-1)

    def complex_pair_to_quat(self, alpha: Tensor, beta: Tensor) -> Tensor:
        return stack([stack([alpha, -beta], dim=-1), stack([beta.conj(), alpha.conj()], dim=-1)], dim=-2)

    def quat_to_complex_pair(self, quat: Tensor) -> Tuple[Tensor, Tensor]:
        return quat[..., 0, 0], -quat[..., 0, 1]

    def real_inv(self, a: Tensor) -> Tensor:
        alpha = (1. + 0j) * a
        beta = zeros_like(alpha)

        return self.complex_pair_to_quat(alpha, beta)

    def imag_inv(self, u: Tensor) -> Tensor:
        alpha = 1j * u[..., -1]
        beta = u[..., 0] + 1j * u[..., 1]

        return self.complex_pair_to_quat(alpha, beta)

    def areas_to_face_mass_matrix_inv(self, As: Tensor):
        values = (1. + 0j) / As.repeat_interleave(2)
        M_F_inv = spdiags(values, tensor(0), (2 * self.num_faces, 2 * self.num_faces))

        return M_F_inv

    def halfedge_quats_to_dirac_operator(self, u_quats: Tensor):
        values = -u_quats.flatten(start_dim=-3) / 2

        face_idxs = arange(self.num_faces).repeat_interleave(12)
        row_idxs = 2 * face_idxs + tensor([0, 0, 1, 1]).repeat(self.num_halfedges)

        vertex_idxs = cat([self.faces[:, 2:], self.faces[:, :2]], dim=-1).reshape(-1).repeat_interleave(4)
        col_idxs = 2 * vertex_idxs + tensor([0, 1, 0, 1]).repeat(self.num_halfedges)

        D = sparse_coo_tensor(stack([row_idxs, col_idxs]), values, (2 * self.num_faces, 2 * self.num_vertices))
        return D

    def vertex_values_and_face_areas_to_integral_operator(self, rhos: Tensor, As: Tensor):
        rhos_by_faces = rhos[self.faces]
        values = (As.unsqueeze(-1) * (rhos_by_faces.sum(dim=-1, keepdims=True) + rhos_by_faces)) / 12
        values = (1. + 0j) * values.flatten(start_dim=-2)

        face_idxs = arange(self.num_faces).repeat_interleave(3)
        vertex_idxs = self.faces.reshape(-1)
        idxs = stack([face_idxs, vertex_idxs])

        R = sparse_coo_tensor(cat([2 * idxs, 2 * idxs + 1], dim=-1), cat([values, values], dim=-1), (2 * self.num_faces, 2 * self.num_vertices))
        return R

    def spin_and_halfedge_quats_to_spun_halfedge_quats(self, lambda_quats: Tensor, u_quats: Tensor) -> Tensor:
        tail_lambda_quats = lambda_quats[self.tail_vertex_idxs]
        tip_lambda_quats = lambda_quats[self.tip_vertex_idxs]

        u_tail_lambda_quats = u_quats @ tail_lambda_quats
        u_tip_lambda_quats = u_quats @ tip_lambda_quats
        new_u_quats = tail_lambda_quats.mH @ (u_tail_lambda_quats / 3 + u_tip_lambda_quats / 6) + tip_lambda_quats.mH @ (u_tail_lambda_quats / 6 + u_tip_lambda_quats / 3)

        return new_u_quats

    def embedding_and_vertex_values_to_spun_embedding(self, fs: Tensor, rhos: Tensor, num_iters: int, verbose=False) -> Tuple[Tensor, Tensor, Tensor]:
        us = self.embedding_to_halfedge_vectors(fs)
        face_areas = self.halfedge_vectors_to_areas(us)
        vertex_areas = self.areas_to_vertex_areas(face_areas).repeat_interleave(2, dim=-1).unsqueeze(-1)
        sqrt_vertex_areas = (1. + 0j) * sqrt(vertex_areas)
        vertex_areas = (1. + 0j) * vertex_areas
        u_quats = self.imag_inv(us)

        D = self.halfedge_quats_to_dirac_operator(u_quats)
        R = self.vertex_values_and_face_areas_to_integral_operator(rhos, face_areas)
        M_F_inv = self.areas_to_face_mass_matrix_inv(face_areas)
        A = D - R
        A_quad = A.H @ (M_F_inv @ A)

        lambdas = randn(4, self.num_vertices, dtype=self.dtype)
        alphas, betas = lambdas[:2] + 1j * lambdas[2:]
        lambdas = self.complex_pair_to_quat(alphas, betas).flatten(start_dim=-3, end_dim=-2)
        for iteration in range(num_iters):
            next_lambdas = sparse_solve_c4t(A_quad, vertex_areas * lambdas)
            lin_solve_err = norm(A_quad @ next_lambdas - vertex_areas * lambdas)

            next_lambdas = qr(sqrt_vertex_areas * next_lambdas).Q / sqrt_vertex_areas
            lambdas = next_lambdas

            grammian = lambdas.mH @ (vertex_areas * lambdas)
            herm_err = norm(grammian - eye(len(grammian), dtype=grammian.dtype))

            eigenvalues = diag(lambdas.mH @ (A_quad @ lambdas)).real
            eigenvalue_err = norm(eigenvalues[..., 0] - eigenvalues[..., 1])
            eigenvalue = eigenvalues[..., 0]

            residual = A_quad @ lambdas - eigenvalues * (vertex_areas * lambdas)

            if verbose:
                print(iteration, lin_solve_err.item(), herm_err.item(), eigenvalue_err.item(), norm(residual).item(), eigenvalue)

        lambda_quats = lambdas.unflatten(-2, (-1, 2))
        new_u_quats = self.spin_and_halfedge_quats_to_spun_halfedge_quats(lambda_quats, u_quats)
        new_us = self.imag(new_u_quats)

        if verbose:
            print(norm(new_us[..., 0::2] + new_us[..., 1::2] + new_us[..., 2::2], dim=-1).max().item())

        alphas = self.halfedge_vectors_to_angles(us)
        L = self.angles_to_laplacian(alphas)

        cot_alphas = 1 / tan(alphas)
        edge_length_ratios = (cot_alphas + self.halfedge_to_twin @ cot_alphas) / 2
        div_new_us = self.tail_vertices_to_halfedges.T @ (edge_length_ratios.unsqueeze(-1) * new_us)

        new_fs = sparse_solve_c4t(L, div_new_us)
        if verbose:
            print(norm(L @ new_fs - div_new_us).item())
            print(norm(self.embedding_to_halfedge_vectors(new_fs) - new_us, dim=-1).max().item())

        return new_fs, lambda_quats, eigenvalue
