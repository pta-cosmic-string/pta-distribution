import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def g(z): 
    return np.real(- np.exp(-z) * sp.special.exp1(-z))

N = 1000
l = 40
x = np.linspace(-l, l, N)
y = np.linspace(-l, l, N)
X, Y = np.meshgrid(x,y)
Z = X + 1j*Y
G0 = g(Z)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, G0)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_zlabel('Re[exp(-z) Ei(z)]')
plt.show()
# psi = 0
# a = 0
# K = np.linspace(0.0001, l, N)
# # z = np.exp(1j*psi) 

# # s = 100
# plt.plot(K, norm(K, a=-1), label=f"true")
# # plt.plot(K, g_app_1(Z), label=f"approx | z={z.real:.2f}+j{z.imag:.2f}")
# # S = [0.01, 0.1, 1, 10, 100, 1000, 10000]
# # for s in S:
# #     plt.plot(T, np.log10(np.abs(g(T, s=s) - g_app(T, s=s))), label=f"delta | s={s}")
# # for s in S:
# #     plt.plot(T, np.log10(np.abs(g(T, s=-s) - g_app(T, s=-s))), label=f"delta | s={-s}")
# # plt.legend()
# plt.show()