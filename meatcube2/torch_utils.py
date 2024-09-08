import torch
from typing import List

def pop_index(matrix: torch.Tensor, i, dim=0) -> torch.Tensor:
    dim = (dim + matrix.dim()) % matrix.dim()
    i = (i + matrix.size(dim)) % matrix.size(dim)
    if i==0:
        slice_base = [slice(None)]*dim
        m = matrix[slice_base+[slice(i+1,None)]]
        m_excluded = matrix[slice_base+[int(i)]]
        return m, m_excluded
    elif i==matrix.size(dim)-1:
        slice_base = [slice(None)]*dim
        m = matrix[slice_base+[slice(None,i)]]
        m_excluded = matrix[slice_base+[int(i)]]
        return m, m_excluded
    else:
        slice_base = [slice(None)]*dim # slices that stay the same [:, :, ...]
        slice_start = slice_base + [slice(i)] # values before the value to remove
        slice_excluded = slice_base + [int(i)] # value to remove
        slice_end = slice_base + [slice(i+1, None)] # values after the value to remove
        m_start = matrix[slice_start]
        m_excluded = matrix[slice_excluded]
        m_end = matrix[slice_end]
        m = torch.cat([m_start, m_end], dim=dim)
        
        return m, m_excluded

def remove_index(matrix: torch.Tensor, i, dims: List[int]) -> torch.Tensor:
    for dim in dims:
        matrix, m_excluded = pop_index(matrix, i, dim=dim)
    return matrix

def append_symmetric(symmetric_matrix: torch.Tensor, vect: torch.Tensor, cell: torch.Tensor):
    vect = vect.view(-1)
    cell = cell.view(-1)

    # from [n, n] to [n, n+1]
    symmetric_matrix = torch.cat([symmetric_matrix, vect.view(-1, 1)], dim=-1)
     # from [n] to [n+1]: add the symmetric component of the vector where the diagonal will be
    vect = torch.cat([vect, cell], dim=-1)
    # from [n, n+1] to [n+1, n+1]
    symmetric_matrix = torch.cat([symmetric_matrix, vect.view(1, -1)], dim=-2)
    
    return symmetric_matrix