# Lab 7 Neural ODE

## Ref
- [DS - Dynamical Systems & Neural ODEs, a good introduction](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Dynamical_systems/dynamical_systems_neural_odes.html)

- [PyTorch Implementation of Differentiable ODE Solvers](https://github.com/rtqichen/torchdiffeq/tree/master)

- [Matlab: Dynamical System Modeling Using Neural ODE](https://www.mathworks.com/help/deeplearning/ug/dynamical-system-modeling-using-neural-ode.html)

- [中文介绍](https://zhuanlan.zhihu.com/p/554790455)

- [NODE（神经常微分方程）介绍, 可以用来预测和VAE](https://juejin.cn/post/7151320014975401991)

Tools that may be useful to your framewkork:
-[Autograd](https://github.com/HIPS/autograd)

## ODE
Consider a simple ODE $\vec{x}'=A\vec{x}$:

```python
# ODE, x' = Ax
A = [[-2, 1], [-4, 1]]
A = np.array(A)

lam, X = np.linalg.eig(A)
print('lambda =', lam)
print('X =', X)
v1 = X[:,0]
v2 = X[:,1]

print(v1, A@v1, lam[0] * v1)
print(v2, A@v2, lam[1] * v2)

# v(t) = v1*exp(lambda1 * t) + v2*exp(lambda2 * t)
dt = 0.1
t = np.arange(0, 6, dt)
num = len(t)
T = np.zeros([2, num])
for i in range(num):
    T[:, i] = v1 * np.exp(lam[0]*t[i]) + v2 * np.exp(lam[1]*t[i])

T = np.abs(T)

plt.figure()
plt.plot(T[0, :], T[1, :], '.')
```
