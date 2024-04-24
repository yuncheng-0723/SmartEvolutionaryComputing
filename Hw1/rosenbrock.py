import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock 函數
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# 生成 x 和 y 的值
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
x1, x2 = np.meshgrid(x1, x2)

# 計算每個點的 Rosenbrock 函數值
z = rosenbrock(x1, x2)

plt.figure()
contour = plt.contour(x1, x2, z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Rosenbrock Function Value')
plt.xlabel('X')
plt.ylabel('X2')
plt.title('2D Rosenbrock Function Contour Plot')
plt.show()