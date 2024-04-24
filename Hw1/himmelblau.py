import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d 

def himmelblau(x,y):
       return (((x**2+y-11)**2) + (((x+y**2-7)**2)))

X=np.linspace(-5,5)
Y=np.linspace(-5,5)

x,y=np.meshgrid(X,Y)
F=himmelblau(x,y)

fig =plt.figure(figsize=(9,9))
ax=plt.axes(projection='3d')
ax.contour3D(x,y,F,450)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F')
ax.set_title('Himmelblau Function')
ax.view_init(50,50)

#plt.contour(x,y,F,15)
plt.show()
# plt.savefig('himmelblau.png')