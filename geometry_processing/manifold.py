from torch import arange, arccos, argsort, cat, cos, diag, diff, eye, float32, IntTensor, maximum, minimum, ones, pi, randn, sin, sort, sparse_coo_tensor, sqrt, stack, Tensor, tensor, zeros_like
from torch.linalg import cross, norm, qr, svd
from torch.sparse import spdiags
from torchsparsegradutils.cupy import sparse_solve_c4t
from typing import Tuple


class Manifold:
    def __init__(self, faces: IntTensor, dtype=float32):
        assert len(faces.shape) == 2
        assert len(faces) > 0
        assert faces.shape[-1] == 3
        assert faces.min() == 0
        assert diff(sort(faces.reshape(-1)).values).max() == 1

        self.faces = faces
        self.dtype = dtype

        self.num_faces = len(faces)
        self.num_halfedges = 3 * self.num_faces
        self.num_vertices = faces.max().item() + 1

        # Halfedge selectors
        idxs = arange(self.num_halfedges).reshape(-1, 3)
        self.jk_idxs = cat([idxs[:, 1:], idxs[:, :1]], dim=-1).reshape(-1)
        self.ki_idxs = cat([idxs[:, 2:], idxs[:, :2]], dim=-1).reshape(-1)

        # Vertices to halfedges
        halfedge_idxs = arange(self.num_halfedges)
        self.tail_vertex_idxs = faces.reshape(-1)
        self.tip_vertex_idxs = cat([faces[:, 1:], faces[:, :1]], dim=-1).reshape(-1)
        values = ones(self.num_halfedges, dtype=self.dtype)
        self.tail_vertices_to_halfedges = sparse_coo_tensor(stack([halfedge_idxs, self.tail_vertex_idxs]), values, (self.num_halfedges, self.num_vertices))
        self.tip_vertices_to_halfedges = sparse_coo_tensor(stack([halfedge_idxs, self.tip_vertex_idxs]), values, (self.num_halfedges, self.num_vertices))
        self.signed_vertices_to_halfedges = self.tip_vertices_to_halfedges - self.tail_vertices_to_halfedges

        pair_idxs = stack([self.tail_vertex_idxs, self.tip_vertex_idxs], dim=-1)
        halfedge_idxs = argsort(self.tail_vertex_idxs)
        pair_idxs = pair_idxs[halfedge_idxs]
        tail_idx_diffs = diff(pair_idxs[:, 0], prepend=tensor([-1]))
        assert tail_idx_diffs.min() == 0 and tail_idx_diffs.max() == 1
        start_idxs = arange(self.num_halfedges)[tail_idx_diffs == 1]
        degrees = diff(start_idxs, append=tensor([self.num_halfedges]))

        twin_idxs = -ones(self.num_halfedges)
        for i, (tail_idx, tip_idx) in enumerate(pair_idxs):
            start_idx = start_idxs[tip_idx]
            degree = degrees[tip_idx]
            neighbor_neighbors = pair_idxs[start_idx:(start_idx + degree), -1]
            reflexive_idxs = arange(degree)[neighbor_neighbors == tail_idx]
            assert len(reflexive_idxs) <= 1
            if len(reflexive_idxs) == 1:
                reflexive_idx = reflexive_idxs[0]
                twin_idx = halfedge_idxs[start_idx + reflexive_idx]
                twin_idxs[i] = twin_idx

        has_twin = twin_idxs > -1
        idxs = stack([halfedge_idxs[has_twin], twin_idxs[has_twin]])
        values = ones(idxs.shape[-1], dtype=self.dtype)
        self.halfedge_to_twin = sparse_coo_tensor(idxs, values, (self.num_halfedges, self.num_halfedges))

    def check_metric(self, ls: Tensor) -> bool:
        is_nonneg = (ls >= 0).all(dim=-1)

        l_ijs = ls[..., ::3]
        l_jks = ls[..., 1::3]
        l_kis = ls[..., 2::3]
        sats_tri_ineq = (l_ijs <= l_jks + l_kis).all(dim=-1) * (l_jks <= l_kis + l_ijs).all(dim=-1) * (l_kis <= l_ijs + l_jks).all(dim=-1)

        return is_nonneg * sats_tri_ineq

    def angles_to_angle_sums(self, alphas: Tensor) -> Tensor:
        return self.tail_vertices_to_halfedges.T @ alphas[..., self.jk_idxs]

    def angles_to_laplacian(self, alphas: Tensor):
        sin_alphas = sin(alphas)
        is_valid_angle = sin_alphas != 0.
        cot_alphas = zeros_like(alphas)
        cot_alphas[is_valid_angle] = cos(alphas[is_valid_angle]) / sin_alphas[is_valid_angle]

        off_diag_L = sparse_coo_tensor(stack([self.tail_vertex_idxs, self.tip_vertex_idxs]), cot_alphas / 2, (self.num_vertices, self.num_vertices))
        off_diag_L = off_diag_L + off_diag_L.T
        diag_L = spdiags(off_diag_L.sum(dim=-1).to_dense(), tensor(0), (self.num_vertices, self.num_vertices))
        L = (off_diag_L - diag_L).coalesce()

        return L

    def areas_to_diag_mass_matrix(self, As: Tensor):
        return spdiags(self.areas_to_vertex_areas(As), tensor(0), (self.num_vertices, self.num_vertices))

    def areas_to_mass_matrix(self, As: Tensor):
        row_idxs = self.faces.unsqueeze(-1).repeat_interleave(3, dim=-1).reshape(-1)
        col_idxs = self.faces.unsqueeze(-2).repeat_interleave(3, dim=-2).reshape(-1)
        idxs = stack([row_idxs, col_idxs])

        template = (ones(3, 3, dtype=self.dtype) + eye(3, dtype=self.dtype)).reshape(-1) / 12
        values = As.repeat_interleave(9) * template.repeat(self.num_faces)

        return sparse_coo_tensor(idxs, values, (self.num_vertices, self.num_vertices))

    def areas_to_vertex_areas(self, As: Tensor) -> Tensor:
        As = As.repeat_interleave(3, dim=-1)
        As_by_vertex = self.tail_vertices_to_halfedges.T @ As / 3

        return As_by_vertex

    def embedding_to_halfedge_vectors(self, fs: Tensor) -> Tensor:
        return self.signed_vertices_to_halfedges @ fs

    def embedding_to_angles(self, fs: Tensor) -> Tensor:
        return self.halfedge_vectors_to_angles(self.embedding_to_halfedge_vectors(fs))

    def embedding_to_angle_sums(self, fs: Tensor) -> Tensor:
        return self.angles_to_angle_sums(self.embedding_to_angles(fs))

    def embedding_to_areas(self, fs: Tensor) -> Tensor:
        return self.halfedge_vectors_to_areas(self.embedding_to_halfedge_vectors(fs))

    def embedding_to_face_normals(self, fs: Tensor, keep_scale: bool = False) -> Tensor:
        return self.halfedge_vectors_to_face_normals(self.embedding_to_halfedge_vectors(fs), keep_scale)

    def embedding_to_metric(self, fs: Tensor) -> Tensor:
        return self.halfedge_vectors_to_metric(self.embedding_to_halfedge_vectors(fs))

    def embedding_to_poisson(self, fs: Tensor, using_diag_mass: bool = False):
        us = self.embedding_to_halfedge_vectors(fs)
        L = self.angles_to_laplacian(self.halfedge_vectors_to_angles(us))

        As = self.halfedge_vectors_to_areas(us)
        if using_diag_mass:
            M = self.areas_to_diag_mass_matrix(As)
        else:
            M = self.areas_to_mass_matrix(As)

        return L, M

    def embedding_and_vertex_values_to_face_grads(self, fs: Tensor, phis: Tensor) -> Tensor:
        us = self.embedding_to_halfedge_vectors(fs)
        Ns = self.halfedge_vectors_to_face_normals(us, keep_scale=True)
        As = norm(Ns, dim=-1)

        Ns = Ns.repeat_interleave(3, dim=-2)
        As = As.repeat_interleave(3, dim=-1).unsqueeze(-1)
        basis_grads = cross(Ns, us) / (2 * As)
        basis_grads = basis_grads[self.jk_idxs]
        basis_grads = basis_grads.unflatten(-2, (self.num_faces, 3))
        grad_phis = (phis[self.faces].unsqueeze(-1) * basis_grads).sum(dim=-2)

        return grad_phis

    def embedding_to_vertex_areas(self, fs: Tensor) -> Tensor:
        return self.areas_to_vertex_areas(self.embedding_to_areas(fs))

    def embedding_to_vertex_mean_curvatures(self, fs: Tensor) -> Tensor:
        return self.halfedge_vectors_to_vertex_mean_curvatures(self.embedding_to_halfedge_vectors(fs))

    def embeddings_to_rotation(self, fs: Tensor, target_fs: Tensor) -> Tensor:
        X = fs - fs.mean(dim=-2, keepdims=True)
        Y = target_fs - target_fs.mean(dim=-2, keepdims=True)

        X /= sqrt(self.embedding_to_areas(X).sum())
        Y /= sqrt(self.embedding_to_areas(Y).sum())

        U, _, V_T = svd(Y.T @ X)
        R = U @ V_T

        return R

    def halfedge_vectors_to_angles(self, us: Tensor) -> Tensor:
        u_jks = us[..., self.jk_idxs, :]
        u_kis = us[..., self.ki_idxs, :]
        cos_alphas = (-u_jks * u_kis).sum(dim=-1) / (norm(u_jks, dim=-1) * norm(u_kis, dim=-1))
        cos_alphas = minimum(maximum(cos_alphas, tensor(-1., dtype=self.dtype)), tensor(1., dtype=self.dtype))
        alphas = arccos(cos_alphas)

        return alphas

    def halfedge_vectors_to_areas(self, us: Tensor) -> Tensor:
        Ns = self.halfedge_vectors_to_face_normals(us, keep_scale=True)

        return norm(Ns, dim=-1) / 2

    def halfedge_vectors_to_face_normals(self, us: Tensor, keep_scale: bool = False) -> Tensor:
        u_ijs = us[..., ::3, :]
        u_jks = us[..., 1::3, :]
        Ns = cross(u_ijs, u_jks, dim=-1)

        return Ns if keep_scale else Ns / norm(Ns, dim=-1, keepdims=True)

    def halfedge_vectors_to_metric(self, us: Tensor) -> Tensor:
        return norm(us, dim=-1)

    def halfedge_vectors_to_vertex_mean_curvatures(self, us: Tensor) -> Tensor:
        Ns = self.halfedge_vectors_to_face_normals(us).repeat_interleave(3, dim=-2)
        twin_Ns = self.halfedge_to_twin @ Ns

        cos_phis = (Ns * twin_Ns).sum(dim=-1)
        cos_phis = minimum(maximum(cos_phis, tensor(-1., dtype=self.dtype)), tensor(1., dtype=self.dtype))
        phis = arccos(cos_phis)

        crosses = cross(Ns, twin_Ns, dim=-1)
        signed_ls = (us * crosses / norm(crosses, dim=-1, keepdims=True)).sum(dim=-1)
        Hs_by_halfedge = signed_ls * phis / 2
        Hs = self.tail_vertices_to_halfedges.T @ Hs_by_halfedge / 2
        return Hs

    def laplacian_and_vertex_areas_to_eigs(self, L, vertex_As: Tensor, num_eigs: int, num_iters: int, verbose: bool = False, eps: float = 0) -> Tuple[Tensor, Tensor]:
        A = -L + eps * spdiags(ones(self.num_vertices, dtype=self.dtype), tensor(0), shape=(self.num_vertices, self.num_vertices))
        vertex_As = vertex_As.unsqueeze(-1)
        sqrt_vertex_As = sqrt(vertex_As)

        eigvecs = randn(self.num_vertices, num_eigs, dtype=self.dtype)
        for iteration in range(num_iters):
            new_eigvecs = sparse_solve_c4t(A, vertex_As * eigvecs)
            lin_solve_err = norm(A @ new_eigvecs - vertex_As * eigvecs)
            eigvecs = new_eigvecs

            eigvecs = qr(sqrt_vertex_As * eigvecs).Q / sqrt_vertex_As
            grammian = eigvecs.T @ (vertex_As * eigvecs)
            grammian_err = norm(grammian - eye(num_eigs, dtype=self.dtype))

            eigvals = diag((A @ eigvecs).T @ eigvecs)
            residuals = A @ eigvecs - eigvals.unsqueeze(-2) * (vertex_As * eigvecs)
            residual_errs = norm(residuals, dim=-2)

            if verbose:
                print(iteration, lin_solve_err.item(), grammian_err.item(), residual_errs)

        return eigvals, eigvecs

    def metric_to_angles(self, ls: Tensor, keep_safe: bool = True) -> Tensor:
        # Compute angles with law of cosines

        if keep_safe:
            assert self.check_metric(ls).all()

        l_ijs = ls
        l_jks = ls[..., self.jk_idxs]
        l_kis = ls[..., self.ki_idxs]
        cos_alphas = (l_jks ** 2 + l_kis ** 2 - l_ijs ** 2) / (2 * l_jks * l_kis)

        if keep_safe:
            cos_alphas = minimum(maximum(cos_alphas, tensor(-1., dtype=self.dtype)), tensor(1., dtype=self.dtype))
            alphas = arccos(cos_alphas)

        else:
            # Assign an angle of pi for all corners across from edges that are too short
            cos_alpha_ks = cos_alphas[..., ::3]
            cos_alpha_is = cos_alphas[..., 1::3]
            cos_alpha_js = cos_alphas[..., 2::3]
            cos_alphas_stacked = stack([cos_alpha_ks, cos_alpha_is, cos_alpha_js], dim=-1)
            is_violating = (cos_alphas_stacked < -1.).any(dim=-1)
            replacements = zeros_like(cos_alphas_stacked[is_violating])
            replacements[(cos_alphas_stacked < -1.)[is_violating]] = pi

            alphas_stacked = zeros_like(cos_alphas_stacked)
            alphas_stacked[~is_violating] = arccos(cos_alphas_stacked[~is_violating])
            alphas_stacked[is_violating] = replacements
            alphas = alphas_stacked.flatten(start_dim=-2)

        return alphas

    def metric_to_areas(self, ls: Tensor) -> Tensor:
        # Compute face areas with Heron's formula
        l_ijs = ls[..., ::3]
        l_jks = ls[..., 1::3]
        l_kis = ls[..., 2::3]
        s_ijks = (l_ijs + l_jks + l_kis) / 2

        return sqrt(s_ijks * (s_ijks - l_ijs) * (s_ijks - l_jks) * (s_ijks - l_kis))

    def metric_to_poisson(self, ls: Tensor, using_diag_mass: bool = False):
        L = self.angles_to_laplacian(self.metric_to_angles(ls))

        As = self.metric_to_areas(ls)
        if using_diag_mass:
            M = self.areas_to_diag_mass_matrix(As)
        else:
            M = self.areas_to_mass_matrix(As)

        return L, M
