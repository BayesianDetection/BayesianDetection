import torch


def torch_diff(x):
    return x[1:] - x[:-1]

def torch_wasserstein_distance(u_values, v_values):
    return _cdf_distance(u_values, v_values)


def _cdf_distance(u_values, v_values):

    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)
    
    all_values = torch.cat((u_values, v_values))
    all_values, _ = torch.sort(all_values)
    # Compute the differences between pairs of successive values of u and v.
    deltas = torch_diff(all_values)
    
    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    # u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    # v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    
    u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], right=True)
    v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], right=True)
    # Calculate the CDFs of u and v using their weights, if specified.

    u_cdf = u_cdf_indices / u_values.shape[0]

    v_cdf = v_cdf_indices / v_values.shape[0]

    return torch.sum(torch.mul(torch.abs(u_cdf - v_cdf), deltas))


# import numpy as np
# from scipy import stats
# size = 100
# u_values=torch.rand(size).cuda()
# v_values=torch.rand(size).cuda()

# vec_1 = u_values.detach().cpu().numpy()
# vec_2 = v_values.detach().cpu().numpy()

# dist = stats.wasserstein_distance(vec_1, vec_2, )

# dist2 = torch_wasserstein_distance(u_values,v_values)
# print(dist)
# print(dist2)


# def check_close(a,b):
#     return np.allclose(a.detach().cpu().numpy(),b)    