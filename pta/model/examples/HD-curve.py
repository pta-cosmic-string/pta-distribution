import numpy as np

PI = np.pi

def mu_u(gamma, N = 100):
    N_ph, N_th = N, N

    p1 = np.array([0, 0, 1])
    p2 = np.array([np.sin(gamma), 0, np.cos(gamma)])
    alpha_1, alpha_2 = 100, 100

    cos_gamma = np.einsum('i,i->',p1,p2)/(np.einsum('i,i->',p1,p1)*np.einsum('i,i->',p2,p2))
    gamma = np.arccos(cos_gamma)

    dphi, dtheta = 2*PI/(N_ph -1), PI/(N_th -1)
    phi = np.linspace(dphi/2, 2*PI - dphi/2, N_ph)
    theta = np.linspace(dtheta/2, PI - dtheta/2, N_th)

    phi, theta = np.meshgrid(phi, theta, indexing='xy')
    dphi, dtheta = np.full(phi.shape, dphi), np.full(theta.shape, dtheta)


    Omega = np.array([-np.sin(theta)*np.cos(phi), -np.sin(theta)*np.sin(phi), -np.cos(theta)])
    dOmega = dphi * np.sin(theta) * dtheta

    m = np.array([np.sin(phi), - np.cos(phi), np.zeros(phi.shape)])
    n = np.array([-np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)])

    e_p = np.einsum('ijk,ljk->iljk', m, m) - np.einsum('ijk,ljk->iljk', n, n) 
    e_c = np.einsum('ijk,ljk->iljk', m, n) + np.einsum('ijk,ljk->iljk', n, m) 
    e = e_p + 1j * e_c
    F_1 = 1/2 * np.einsum('il,iljk->jk',np.einsum('i,l->il', p1, p1), e) / (1 + np.einsum('ijk,i->jk', Omega, p1)) 
    F_2 = 1/2 * np.einsum('il,iljk->jk',np.einsum('i,l->il', p2, p2), e) / (1 + np.einsum('ijk,i->jk', Omega, p2)) 

    T_1 =  1 - np.exp(-1j  * 2 * PI * alpha_1 * (1 + np.einsum('ijk,i->jk', Omega, p1)))
    T_2 =  1 - np.exp(-1j  * 2 * PI * alpha_2 * (1 + np.einsum('ijk,i->jk', Omega, p2)))

    R_1 = F_1 * T_1
    R_2 = F_2 * T_2

    mu = 1/(4*PI) * np.real(np.sum(R_1 * np.conjugate(R_2) * dOmega))
    
    return mu

def mu_0(gamma):
    cos_gamma = np.cos(gamma)
    mu = 1/3 - 1/6 * (1 - cos_gamma)/2 + (1 - cos_gamma)/2 * np.log((1 - cos_gamma)/2)
    return mu

gamma = np.linspace(0.0001, PI, 100)

for g in gamma:
    m_u = mu_u(g,N=1000)
    m_0 = mu_0(g)
    err = abs(((m_u - m_0)/m_0))* 100
    print(f"err mu = {err:.2f}%")

