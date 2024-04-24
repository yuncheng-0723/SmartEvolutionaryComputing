import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def eggholder(x, y) :
    return (- (y + 47) * np.sin( np.sqrt( np.abs( x / 2 + y + 47 ) ) )
                - x * np.sin( np.sqrt( np.abs( x - ( y + 47 ) ) ) ))

plt.style.use('dark_background')
plt.rcParams.update({'axes.labelsize':15, 'axes.titlesize':15})

# 3d
fig = plt
ax = fig.figure().add_subplot(311, projection='3d')

axis = np.linspace(-1000, 1000, 100)
X, Y = np.meshgrid(axis, axis)
z = eggholder(X, Y)

ax.plot_surface(X, Y, z, cmap='magma')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.view_init(50,50)

fig.show()

# 2d
plt.figure()
contour = plt.contour(X, Y, z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Rosenbrock Function Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Eggholder Function Contour Plot')
plt.show()
plt.savefig('eggholder.png')