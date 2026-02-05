"""CKA and related functions."""
# Author: Christian Ferreyra
# Date: 08/2025

import math

import torch


def to_vector_torch(x, kdiag=False, upper=False):
    """Flatten the triangular part of a square matrix into a 1D vector.

    This converts a symmetric (or general) square matrix into a compact vector
    representation by selecting either the lower- or upper-triangular entries.

    Parameters
    ----------
    x : torch.Tensor
        A tensor whose last two dimensions are square matrices (..., N, N).
        Note: the indexing used returns values from the last two dims only.
    kdiag : bool, optional
        Whether to keep the diagonal in the output.
        - If True, includes diagonal elements.
        - If False, excludes diagonal elements.
        Default is False.
    upper : bool, optional
        Whether to extract the upper triangle.
        - If True, use upper-triangular indices (including/excluding diag via kdiag).
        - If False, use lower-triangular indices.
        Default is False.

    Returns
    -------
    torch.Tensor
        A tensor containing the selected triangular entries of `x`.
        The output is indexed by triangular coordinates and therefore is flattened
        over the last two dimensions.

    Notes
    -----
    - `torch.tril_indices` / `torch.triu_indices` are used to generate coordinates.
    - the `offset` controls whether the diagonal is included:
        offset = 0 includes diagonal
        offset = 1 (upper) or -1 (lower) excludes diagonal
    """
    n_dim = x.shape[-1]
    ind_fn = torch.triu_indices if upper else torch.tril_indices
    offset = int(not kdiag)
    offset *= 1 if upper else -1
    indices = ind_fn(n_dim, n_dim, offset=offset)
    return x[indices]


def to_matrix_torch(x, tril=True, diag=False):
    """Convert a vector of triangular entries into a symmetric square matrix.

    The function interprets the first dimension of `x` as containing the entries
    of a triangular portion of a matrix (lower if `tril=True`, upper otherwise),
    and then reconstructs a symmetric matrix by mirroring those entries across
    the diagonal.

    Parameters
    ----------
    x : torch.Tensor
        A tensor whose first dimension indexes the triangular entries.
        Shape is (M, ...) where M is the number of triangular elements.
        Any remaining dimensions (...) are treated as batch/feature dims and are
        preserved in the reconstructed matrix.
    tril : bool, optional
        If True, interpret `x` as lower-triangular entries.
        If False, interpret `x` as upper-triangular entries.
        Default is True.
    diag : bool, optional
        If True, interpret `x` as including diagonal entries.
        If False, interpret `x` as excluding diagonal entries.
        Default is False.

    Returns
    -------
    torch.Tensor
        A symmetric tensor of shape (N, N, ...) where N is inferred from M and
        whether the diagonal is included.

    Raises
    ------
    ValueError
        If the length of the input vector does not correspond to a valid
        triangular number given the `diag` setting.

    Notes
    -----
    Let N be the matrix size.
    - if diag=True, M should equal N*(N+1)/2
    - if diag=False, M should equal N*(N-1)/2

    This function infers N from M by solving the quadratic relation.
    """
    # infer number of matrix elements n_elem (called N in the notes)
    # - when diag=True, solve M = N*(N+1)/2
    # - when diag=False, solve M = N*(N-1)/2
    f = -1 if diag else 1
    n_elem = int(f * 0.5 + 0.5 * math.sqrt(1 + 8 * x.shape[0]))

    if len(x) != n_elem * (n_elem + 1) / 2:
        msg = "Input vector size does not correspond to a triangular matrix with a diagonal."
        raise ValueError(msg)

    # build triangular indices depending on whether x is lower or upper,
    # and whether diagonal elements should be present
    if tril:
        indices = torch.tril_indices(n_elem, n_elem, offset=-int(not diag))
    else:
        indices = torch.triu_indices(n_elem, n_elem, offset=int(not diag))

    # allocate output matrix with any extra dims preserved
    size = [n_elem, n_elem]
    if len(x.shape[1:]) != 0:
        size.extend(x.shape[1:])
    x_m = torch.zeros(size=size, dtype=x.dtype, device=x.device)

    # fill both triangles to enforce symmetry
    # note: indexing uses the first dimension of x as the list of entries
    x_m[indices[0], indices[1]] = x[:, ...]
    x_m[indices[1], indices[0]] = x[:, ...]

    return x_m


def double_center(a, vector_in=True, vector_out=True):
    """Double-center a matrix (or vectorized triangular form) by removing row/col means.

    Double-centering transforms a matrix A into:
        A_c = A - mean_row(A) - mean_col(A) + mean_all(A)

    This is commonly used for kernel/Gram matrices to enforce zero-mean structure
    in feature space.

    Parameters
    ----------
    a : torch.Tensor
        Input matrix or vectorized triangular representation.
        - if vector_in=True: `a` is assumed to be a vector encoding a symmetric matrix
          including the diagonal (diag=True convention used in `to_matrix_torch`).
        - if vector_in=False: `a` is assumed to already be a matrix with last two dims
          corresponding to (N, N).
    vector_in : bool, optional
        Whether the input is a vectorized triangular representation. Default is True.
    vector_out : bool, optional
        Whether to return the centered result as a vectorized triangular representation.
        If True, returns the vector including the diagonal (kdiag=True). Default is True.

    Returns
    -------
    torch.Tensor
        The double-centered matrix (if vector_out=False) or its vectorized triangular
        representation (if vector_out=True).

    Notes
    -----
    - when converting vector -> matrix, this uses diag=True, meaning the vector is
      expected to include diagonal entries.
    - when converting matrix -> vector, this uses kdiag=True to keep the diagonal.
    """
    # diag parameters forces the to_matrix and to_vector to consider
    # the diagonal elements or not
    if vector_in:
        a_mat = to_matrix_torch(a, diag=True)
    else:
        a_mat = a

    # A_c = A - row_mean - col_mean + grand_mean
    a_centered = (
        a_mat
        - torch.mean(a_mat, dim=-2, keepdim=True)
        - torch.mean(a_mat, dim=-1, keepdim=True)
        + torch.mean(a_mat, dim=(-2, -1), keepdim=True)
    )

    if vector_out:
        a_centered = to_vector_torch(a_centered, kdiag=True)

    return a_centered


def matrix_unit_norm(x):
    """Normalize matrices to unit Frobenius norm.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with last two dimensions representing matrices (..., N, N).

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as `x`, where each matrix slice is divided by its
        Frobenius norm:
            ||X||_F = sqrt(sum_{i,j} X_{i,j}^2)

    Notes
    -----
    - uses `torch.linalg.norm(..., ord="fro", dim=(-2, -1), keepdim=True)` so that
      broadcasting works correctly when dividing.
    """
    # Ensure dimensions are correct for norm calculation
    return x / torch.linalg.norm(x, ord="fro", dim=(-2, -1), keepdim=True)


def debias_and_normalize_gram(gram, return_mask=False):
    """Remove the diagonal from a Gram matrix and normalize to unit Frobenius norm.

    This is a simple "de-biasing" step often used for CKA variants:
    - set diagonal entries to 0 (self-similarities removed)
    - normalize the resulting matrix by its Frobenius norm

    Parameters
    ----------
    gram : torch.Tensor
        Gram/kernel matrix tensor with shape (..., N, N).
    return_mask : bool, optional
        If True, also return the off-diagonal mask used to zero the diagonal.
        Default is False.

    Returns
    -------
    torch.Tensor
        The debiased and Frobenius-normalized Gram matrix.
    torch.Tensor, optional
        The broadcastable mask (same shape as `gram`) if return_mask=True.

    Notes
    -----
    - the diagonal mask is constructed with `torch.eye(N)` and expanded across any
      leading batch dims so it can be broadcast-multiplied with `gram`.
    - the returned `mask` is 1 on off-diagonal entries and 0 on the diagonal.
    """
    # build an identity matrix on the last two dims, then expand to match gram dims
    mask = torch.eye(gram.shape[-1], device=gram.device, requires_grad=False)

    # reshape mask to match gram's dimensions for broadcasting
    for _ in range(gram.ndim - 2):
        mask = mask.unsqueeze(0)

    # convert identity into "off-diagonal" mask: 1 off-diagonal, 0 on diagonal
    mask = 1 - mask

    # zero out diagonal
    gram = mask * gram

    # normalize after diagonal removal
    gram = matrix_unit_norm(gram)
    if return_mask:
        return gram, mask
    return gram


def compute_cka(x_gram, y_gram, *, remove_diag: bool = False):
    """Compute centered kernel alignment (CKA) between two Gram matrices.

    This computes the normalized Frobenius inner product between two Gram matrices:
        CKA(X, Y) = <Gx, Gy>_F
    assuming the inputs are already appropriately centered and normalized.

    Parameters
    ----------
    x_gram : torch.Tensor
        Gram matrix tensor with shape (..., N, N).
    y_gram : torch.Tensor
        Gram matrix tensor with shape (..., N, N).
    remove_diag : bool, optional
        If True, apply a diagonal-removal + Frobenius normalization to x_gram,
        and apply the same diagonal mask to y_gram before normalizing y_gram.
        Default is False.

    Returns
    -------
    torch.Tensor
        CKA values computed as sum over the last two dimensions, i.e. shape equals
        the broadcasted leading dimensions of the inputs.

    Notes
    -----
    - Unbiased version based on Lange et al., 2023.
    - Code based on: https://github.com/wrongu/repsim
    - This function assumes that if remove_diag=False, the caller has already
      performed any desired centering/debiasing/normalization.
    """
    if remove_diag:
        # debiased version if input grams have not been debiased
        x_gram, mask = debias_and_normalize_gram(x_gram, return_mask=True)

        # apply same diagonal removal to y_gram, then normalize
        y_gram = mask * y_gram
        y_gram = matrix_unit_norm(y_gram)

    # sum over the last two dimensions (matrix dimensions) for Frobenius inner product
    cka = torch.sum(x_gram * y_gram, dim=(-2, -1))
    return cka


def compute_cka_matrix(x_gram, y_gram, *, remove_diag: bool = False):
    """Compute a matrix of CKA scores for all layer pairs.

    This is a batched variant that computes CKA between every pair of layers
    represented in `x_gram` and `y_gram`.

    Parameters
    ----------
    x_gram : torch.Tensor
        If 4D: expected shape (B, L, N, N), where:
            B = batch size (or grouping dimension)
            L = number of layers (or representations) for x
            N = number of samples (Gram size)
        Otherwise: expected shape (L, N, N) or (B, N, N) depending on usage.
    y_gram : torch.Tensor
        If 4D: expected shape (B, T, N, N), where T is number of layers for y.
        Otherwise: expected shape compatible with the non-4D branch.
    remove_diag : bool, optional
        If True, apply diagonal removal + Frobenius normalization as in `compute_cka`.
        Default is False.

    Returns
    -------
    torch.Tensor
        If x_gram.ndim == 4, returns shape (B, L, T), containing CKA for each
        layer pair (l, t) within each batch element.
        Otherwise, returns the result of a matrix multiplication between flattened
        grams, with shape determined by the first dims of x_gram and y_gram.

    Notes
    -----
    - Unbiased version based on Lange et al., 2023.
    - for the 4D case:
        flat_x has shape (B, L, N*N)
        flat_y has shape (B, T, N*N)
        einsum "blk,btk->blt" computes pairwise inner products over k = N*N
    """
    if remove_diag:
        x_gram, mask = debias_and_normalize_gram(x_gram, return_mask=True)
        y_gram = mask * y_gram
        y_gram = matrix_unit_norm(y_gram)

    if x_gram.ndim == 4:
        # flatten matrix dims into a single feature dim for inner products
        flat_x = x_gram.view(*x_gram.shape[:-2], -1)
        flat_y = y_gram.view(*y_gram.shape[:-2], -1)

        # compute inner product between every pair of layers
        # b: batch, l: x layer, t: y layer, k: flattened gram entries
        cka_matrix = torch.einsum("blk,btk->blt", flat_x, flat_y)
    else:
        # non-batched pairwise: treat first dim as list of representations
        # and compute all pairwise inner products after flattening (N, N) -> (N*N)
        cka_matrix = torch.matmul(
            x_gram.view(x_gram.shape[0], -1),
            y_gram.view(y_gram.shape[0], -1).t(),
        )
    return cka_matrix


def compute_angular_cka(cka):
    """Compute Angular CKA from (linear) CKA values.

    Angular CKA maps similarity in [-1, 1] to an angle in [0, pi]:
        angular_cka = arccos(cka)

    Parameters
    ----------
    cka : torch.Tensor
        Tensor of CKA values. Typically in [-1, 1], but may have small numerical
        overshoots due to floating point error.

    Returns
    -------
    torch.Tensor
        Tensor of angles (radians), same shape as `cka`.

    Notes
    -----
    - Unbiased version based on Lange et al., 2023.
    - Code based on: https://github.com/wrongu/repsim
    - clipping avoids NaNs from values slightly outside [-1, 1].
    """
    # Clipping because arccos(1.00000000001) gives NaN
    return torch.arccos(torch.clip(cka, -1.0, 1.0))
