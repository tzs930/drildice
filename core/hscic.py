import torch

def compute_pdist_sq(x, y=None):
    """compute the squared paired distance between x and y."""
    if y is not None:
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        return torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0)
    a = x.view(x.shape[0], -1)
    aTa = torch.mm(a, a.T)
    aTa_diag = torch.diag(aTa)
    aTa = torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0)

    ind = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    aTa[ind[0], ind[1]] = 0
    return aTa + aTa.transpose(0, 1)

def gaussian_kernel(X, sigma2=None, Y=None, normalized=False, **ignored):
    if normalized:
        X = X / torch.linalg.norm(X, dim=1, keepdim=True)
        if Y is not None:
            Y = Y / torch.linalg.norm(Y, dim=1, keepdim=True)
    Dxx = compute_pdist_sq(X, Y)
    if sigma2 is None:
        sigma2 = Dxx.median()
    Kx = torch.exp(-Dxx / sigma2)
    return Kx

def estimate_hscic(X, Y, Z, ridge_lambda=1e-2, use_median=False, normalize_kernel=False, sigma2=None):
    '''X ind. Y | Z '''
    # (1) action regularization version : X = imitator action
    # (2) regularized representation version : X = varphi(Obs)    

    if sigma2 is None:
        if use_median:
            sigma2_ = None
        else:
            sigma2_ = 1.
    else:
        sigma2_ = sigma2

    Kx = gaussian_kernel(X, sigma2=sigma2_, normalized=normalize_kernel)
    Ky = gaussian_kernel(Y, sigma2=sigma2_, normalized=normalize_kernel)
    Kz = gaussian_kernel(Z, sigma2=sigma2_, normalized=normalize_kernel)
    
    n = Kz.shape[0]    
    
    WtKzz = torch.linalg.solve(Kz + ridge_lambda * n * torch.eye(n).to(Kz.device), Kz)     
    term_1 = (WtKzz * ((Kx * Ky) @ WtKzz)).sum()    # tr(WtKzz.T @ (Kx * Ky) @ WtKzz)
    WkKxWk = WtKzz * (Kx @ WtKzz)
    KyWk = Ky @ WtKzz
    term_2 = (WkKxWk * KyWk).sum()        
    term_3 = (WkKxWk.sum(dim=0) * (WtKzz * KyWk).sum(dim=0)).sum()    
    result = (term_1 - 2 * term_2 + term_3) / n    
    
    # W = (Kz + ridge_lambda * n * torch.eye(n)).inverse()
    # A1 = Kz.T @ W @ ( Kx * Ky ) @ W.T @ Kz
    # A2 = Kz.T @ W @ ( (Kx@W.T@Kz) * (Ky@W.T@Kz) )
    # A3 = (Kz.T @ W @ Kx @ W.T @ Kz) * (Kz.T @ W @ Ky @ W.T @ Kz)
    # result2 = (A1-2*A2+A3).trace()/n
    
    # W = (Kz + ridge_lambda  * n * torch.eye(n)).inverse()
    # total_term = 0.
    # for i in range(n):
    #     kz = Kz[i]  #(n,1)
    #     t1 = kz.T @ W @ (Kx * Ky) @ W.T @ kz
    #     t2 = kz.T @ W @ ( (Kx@W.T@kz) * (Ky@W.T@kz) )
    #     t3 = (kz.T @ W @ Kx @ W.T @ kz) * (kz.T @ W @ Ky @ W.T @ kz)
    #     total_term += t1 - 2 * t2 + t3
    # result3 = total_term / n
    
    return result

def hsic_matrices(Kx, Ky, biased=False):
    n = Kx.shape[0]

    if biased:
        a_vec = Kx.mean(dim=0)
        b_vec = Ky.mean(dim=0)
        # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
        return (Kx * Ky).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()

    else:
        tilde_Kx = Kx - torch.diagflat(torch.diag(Kx))
        tilde_Ky = Ky - torch.diagflat(torch.diag(Ky))

        u = tilde_Kx * tilde_Ky
        k_row = tilde_Kx.sum(dim=1)
        l_row = tilde_Ky.sum(dim=1)
        mean_term_1 = u.sum()  # tr(KL)
        mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
        mu_x = tilde_Kx.sum()
        mu_y = tilde_Ky.sum()
        mean_term_3 = mu_x * mu_y

        # Unbiased HISC.
        mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
    return mean

def estimate_hsic(X, Y, Kx_sigma2=1., Ky_sigma2=1., biased=False):
    '''X ind. Y'''
    # todo:
    #  alternative implementation for RFF
    #  biased/unbiased HSIC choice
    #  faster implementation for biased
    Kx = gaussian_kernel(X, sigma2=Kx_sigma2)
    Ky = gaussian_kernel(Y, sigma2=Ky_sigma2)
    
    return hsic_matrices(Kx, Ky, biased)
