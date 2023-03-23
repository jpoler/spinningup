import numpy as np
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def average_kl(pi_old, pi_new):
    kl = torch.distributions.kl.kl_divergence(pi_old, pi_new)
    return kl.mean()

def surrogate_advantage(logp_old, logp_new, adv):
    ratio = torch.exp(logp_new - logp_old)
    return (ratio * adv).mean()

def conjugate_gradients(Ax, x, b, max_iters, eps):
    eps_squared = eps**2
    r = b - Ax(x)
    d = r
    delta_new = torch.dot(r, r)
    delta_0 = delta_new
    for i in range(max_iters):
        if delta_new <= eps_squared * delta_0:
            return x

        q = Ax(d)
        d_dot_q = torch.dot(d, q) + eps
        a = delta_new / d_dot_q
        x += a * d
        if i % 50 == 0:
            r = b - Ax(x)
        else:
            r -= a * q
        delta_old = delta_new
        delta_new =  torch.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d

    return x

def compute_direction(obs, act, adv, pi_loss_g, actor_old, actor_new, cg_iters,  damping_coeff, delta, eps=1e-8):
    pi_old, _ = actor_old(obs, act)
    pi_new, _ = actor_new(obs, act)

    kl = average_kl(pi_old, pi_new)

    def Hx(x):
        grads = torch.autograd.grad(kl, actor_new.gradient_parameters(), create_graph=True)
        g = torch.nn.utils.parameters_to_vector(grads)
        g_dot_x = torch.dot(g, x)
        grads = [g.contiguous() for g in torch.autograd.grad(g_dot_x, actor_new.gradient_parameters(), retain_graph=True)]
        return damping_coeff*x + torch.nn.utils.parameters_to_vector(grads)

    x_hat = conjugate_gradients(Hx, torch.zeros_like(pi_loss_g), pi_loss_g, cg_iters, eps)
    x_hat_coeff = torch.sqrt((2*delta) / (torch.dot(x_hat, pi_loss_g) + eps))

    return x_hat, x_hat_coeff

def line_search(obs, act, adv, d, backtrack_iters, backtrack_coeff, delta, actor_old, actor_new):
    with torch.no_grad():
        theta_old = torch.nn.utils.parameters_to_vector(actor_old.gradient_parameters())
        pi_old, logp_old = actor_old(obs, act)
        for i in range(backtrack_iters):
            theta_new = theta_old + (backtrack_coeff**i) * d
            torch.nn.utils.vector_to_parameters(theta_new, actor_new.gradient_parameters())
            pi_new, logp_new = actor_new(obs, act)
            akl = average_kl(pi_old, pi_new)
            surrogate_adv = surrogate_advantage(logp_old, logp_new, adv)
            if (akl <= delta) and (surrogate_adv > 0):
                return actor_new, akl, surrogate_adv, i

    return actor_old, 0, 0, i
