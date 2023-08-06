import torch


class HawkesProcessDensity(torch.nn.Module):
    def __init__(self, T):
        super(HawkesProcessDensity, self).__init__()
        self.mu = torch.nn.Parameter(torch.ones(1))
        self.a = torch.nn.Parameter(torch.ones(1))
        self.b = torch.nn.Parameter(torch.ones(1))
        self.T = T

    def forward(self, t_n):
        product = torch.tensor(1)
        for i in range(len(t_n)):
            tmp_sum = torch.tensor(0)
            for j in range(i):
                tmp_sum += self._exp_kernel_func(t_n[i] - t_n[j])
            product *= self.mu + tmp_sum

        sum_of_integrations = torch.tensor(0)
        for i in range(len(t_n)):
            tmp_t = torch.arange(t_n[i], self.T, 0.00001)
            tmp_y = self._exp_kernel_func(tmp_t - t_n[i])
            sum_of_integrations = torch.trapezoid(tmp_y, tmp_t)

        term_of_exp = torch.exp(- self.mu * self.T - sum_of_integrations)

        return product * term_of_exp


    def string(self):
        return 'HawkesProcessDensity(mu={}, a={}, b={}, T={})'.format(self.mu, self.a, self.b, self.T)

    def _exp_kernel_func(self, t):
        return self.a * self.b * torch.exp(- self.b * t)

def _calc_logL(t_n, mu, a, b, T):
    mat_t_n = t_n.repeat((t_n.shape[0], 1))
    mat_t_n_T = torch.t(mat_t_n)

    mat_diff_t_n = torch.tril(mat_t_n - mat_t_n_T, diagonal=-1)

    logL = torch.sum(torch.log(mu + torch.sum(a * b * torch.exp(- b * (mat_diff_t_n))))) - (mu * T + torch.sum(a * (1 - torch.exp(- b * (T - t_n)))))
    return logL


def _calc_denominator_dlogL(t_n, mu, a, b):
    mat_t_n = t_n.repeat((t_n.shape[0], 1))
    mat_t_n_T = torch.t(mat_t_n)

    mat_diff_t_n = torch.tril(mat_t_n - mat_t_n_T, diagonal=-1)

    denominator = mu + torch.sum(a * b * torch.exp(- b * (mat_diff_t_n)))
    return denominator


def _calc_dlogL_dmu(t_n, mu, a, b, T):
    denominator = _calc_denominator_dlogL(t_n, mu, a, b)
    dlogL_dmu = torch.sum(1 / denominator) - T
    return dlogL_dmu

def _calc_dlogL_da(t_n, mu, a, b, T):
    denominator = _calc_denominator_dlogL(t_n, mu, a, b)

    mat_t_n = t_n.repeat((t_n.shape[0], 1))
    mat_t_n_T = torch.t(mat_t_n)
    mat_diff_t_n = torch.tril(mat_t_n - mat_t_n_T, diagonal=-1)

    numerator = torch.sum(b * torch.exp(- b * (mat_diff_t_n)))

    bias = torch.sum(1 - torch.exp(- b * (T - t_n)))

    dlogL_da = torch.sum(numerator / denominator) - bias
    return dlogL_da

def _calc_dlogL_db(t_n, mu, a, b, T):
    denominator = _calc_denominator_dlogL(t_n, mu, a, b)

    mat_t_n = t_n.repeat((t_n.shape[0], 1))
    mat_t_n_T = torch.t(mat_t_n)
    mat_diff_t_n = torch.tril(mat_t_n - mat_t_n_T, diagonal=-1)

    numerator = torch.sum(a * torch.exp(- b * (mat_diff_t_n)) * (1 - b * (mat_diff_t_n)))

    bias = torch.sum(a * (T - t_n) * torch.exp(- b * (T - t_n)))

    dlogL_db = torch.sum(numerator / denominator) - bias
    return dlogL_db

class HawkesProcessDensityLogLikelihood(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t_n, mu, a, b, T):
        ctx.save_for_backward(t_n, mu, a, b)
        ctx.T = T

        logL = _calc_logL(t_n, mu, a, b, T)
        return logL
    
    @staticmethod
    def backward(ctx, grad_output):
        t_n, mu, a, b = ctx.saved_tensors
        T = ctx.T
        dlogL_dmu = grad_output * _calc_dlogL_dmu(t_n, mu, a, b, T)
        dlogL_da = grad_output * _calc_dlogL_da(t_n, mu, a, b, T)
        dlogL_db = grad_output * _calc_dlogL_db(t_n, mu, a, b, T)

        # print(t_n, mu, a, b, T)
        # print(dlogL_dmu, dlogL_da, dlogL_db)

        return None, -dlogL_dmu, -dlogL_da, -dlogL_db, None


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    model = HawkesProcessDensity(T=500)
    x = torch.linspace(0, 500, 500)
    # y = torch.randint(0, 50, torch.Size([500]))
    y = torch.ones(torch.Size([500]))
    # peak_idx = torch.randint(1, 500, torch.Size([100]))
    # y[peak_idx] = torch.randint(10, 50, torch.Size([100]), dtype=torch.double)
    # y[peak_idx + 1] = torch.randint(5, 25, torch.Size([100]), dtype=torch.double)
    # y[peak_idx - 1] = torch.randint(5, 25, torch.Size([100]), dtype=torch.double)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    nll_func = HawkesProcessDensityLogLikelihood

    for i in range(50000):
        params = model.parameters()
        
        nll = nll_func.apply(y, *params, 500)
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()

        with torch.no_grad():
            list(model.parameters())[0].clamp_(0)

        if i % 1000 == 0:
            print(f"{i:08}: LogLikelihood: {nll.item()}")

    print(list(model.parameters()))