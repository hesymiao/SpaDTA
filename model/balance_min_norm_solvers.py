from __future__ import annotations

import torch


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1: torch.Tensor, v1v2: torch.Tensor, v2v2: torch.Tensor):
        if v1v2 >= v1v1:
            gamma = v1v1.new_tensor(0.999)
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = v1v1.new_tensor(0.001)
            cost = v2v2
            return gamma, cost
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2.0 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        device = vecs[0][0].device
        dtype = vecs[0][0].dtype
        dmin = torch.tensor(float("inf"), device=device, dtype=dtype)
        sol = None
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = sum(torch.mul(vecs[i][k], vecs[j][k]).sum() for k in range(len(vecs[i])))
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = sum(torch.mul(vecs[i][k], vecs[i][k]).sum() for k in range(len(vecs[i])))
                if (j, j) not in dps:
                    dps[(j, j)] = sum(torch.mul(vecs[j][k], vecs[j][k]).sum() for k in range(len(vecs[j])))
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        if sol is None:
            raise RuntimeError("MinNormSolver 未找到有效的二维初始化解。")
        return sol, dps

    @staticmethod
    def _projection2simplex(y: torch.Tensor) -> torch.Tensor:
        m = y.numel()
        sorted_y, _ = torch.sort(y, descending=True)
        tmpsum = y.new_tensor(0.0)
        tmax_f = (torch.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum = tmpsum + sorted_y[i]
            tmax = (tmpsum - 1.0) / float(i + 1)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.clamp(y - tmax_f, min=0.0)

    @staticmethod
    def _next_point(cur_val: torch.Tensor, grad: torch.Tensor, n: int) -> torch.Tensor:
        proj_grad = grad - (torch.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / proj_grad[proj_grad > 0]

        t = cur_val.new_tensor(1.0)
        if torch.any(tm1 > 1e-7):
            t = torch.min(tm1[tm1 > 1e-7])
        if torch.any(tm2 > 1e-7):
            t = torch.minimum(t, torch.min(tm2[tm2 > 1e-7]))
        next_point = proj_grad * t + cur_val
        return MinNormSolver._projection2simplex(next_point)

    @staticmethod
    def find_min_norm_element(vecs):
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        device = vecs[0][0].device
        dtype = vecs[0][0].dtype
        sol_vec = torch.zeros(n, device=device, dtype=dtype)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1.0 - init_sol[1]
        if n < 3:
            return sol_vec, init_sol[2]

        grad_mat = torch.zeros((n, n), device=device, dtype=dtype)
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        iter_count = 0
        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)

            v1v1 = torch.dot(sol_vec, torch.matmul(grad_mat, sol_vec))
            v1v2 = torch.dot(sol_vec, torch.matmul(grad_mat, new_point))
            v2v2 = torch.dot(new_point, torch.matmul(grad_mat, new_point))

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1.0 - nc) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < MinNormSolver.STOP_CRIT:
                return new_sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1

        return sol_vec, torch.dot(sol_vec, torch.matmul(grad_mat, sol_vec))
