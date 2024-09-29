import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from tqdm import tqdm

gs = 100  # Grid spacing
R = 0.5  # Radius of loop (mm)
wr = 0.1  # Radius of wire (mm)
p = 0.1  # Pitch of wire, center-to-center (mm)
N = 100  # Number of segments in a single loop of wire
n = 5  # Number of loops of wire
theta = np.empty(n * N)
mu = 1  # Magnetic permeability
I = -1  # Current
C = mu * I / (4 * np.pi)
xmin = -2.1
xmax = 2.1
ymin = -2.1
ymax = 2.1
zmin = -1.1
zmax = p * n * 2 + 1.1

z_fixed = 0.5  # Fixed z-plane for the xy plot

# Grid for xy-plane
x = np.linspace(xmin, xmax, gs)  # Positions for x
y = np.linspace(ymin, ymax, gs)  # Positions for y
X, Y = np.meshgrid(x, y, indexing='ij')

# Grid for zy-plane
z = np.linspace(zmin, zmax, gs)  # Positions for z
Y_zy, Z_zy = np.meshgrid(y, z, indexing='ij')

# Arrays for vector potential in xy and zy planes
Ax_xy = np.zeros([gs, gs])  # x components of vector potential in xy-plane
Ay_xy = np.zeros([gs, gs])  # y components of vector potential in xy-plane

Ay_zy = np.zeros([gs, gs])  # y components of vector potential in zy-plane
Az_zy = np.zeros([gs, gs])  # z components of vector potential in zy-plane

norms_xy = np.zeros([gs, gs])  # Norms for xy-plane
norms_zy = np.zeros([gs, gs])  # Norms for zy-plane


# Function to calculate vector potential A
def find_A(pos, theta, R, N, wr):
    dA = 0
    for k in range(1, theta.size):
        rs = np.array([R * np.cos(theta[k] - np.pi / N),
                       R * np.sin(theta[k] - np.pi / N),
                       (p * (theta[k] - np.pi / N)) / np.pi])
        r = pos - rs
        dl = np.array([R * (np.cos(theta[k]) - np.cos(theta[k - 1])),
                       R * (np.sin(theta[k]) - np.sin(theta[k - 1])),
                       p / N])
        
        dA += C * dl / LA.norm(r)
    return dA


# Calculate the vector potential A for both the xy-plane and zy-plane
def find_field():
    # Calculate vector potential in the xy-plane at z_fixed
    for i in tqdm(range(x.size)):
        for j in range(y.size):
            pos_xy = np.array([x[i], y[j], z_fixed])  # Fixed z
            Ax_xy[i, j], Ay_xy[i, j], _ = find_A(pos_xy, theta, R, N, wr)
            norms_xy[i, j] = LA.norm([Ax_xy[i, j], Ay_xy[i, j]])

    # Calculate vector potential in the zy-plane at x=0
    for j in tqdm(range(y.size)):
        for k in range(z.size):
            pos_zy = np.array([0, y[j], z[k]])  # x=0 plane
            _, Ay_zy[j, k], Az_zy[j, k] = find_A(pos_zy, theta, R, N, wr)
            norms_zy[j, k] = LA.norm([Ay_zy[j, k], Az_zy[j, k]])

    return norms_xy, norms_zy


# Plot quiver diagram in the xy-plane at fixed z
def plot_xy_plane():
    fig, ax = plt.subplots(figsize=(20, 16), dpi=600)

    ax.quiver(X, Y, Ax_xy, Ay_xy)  # Quiver plot of the vector potential in the xy-plane

    ax.set_xlim((xmin, xmax))  # Set the xlim to xmin, xmax
    ax.set_ylim((ymin, ymax))  # Set the ylim to ymin, ymax

    plt.xlabel('X axis', fontsize=30)
    plt.ylabel('Y axis', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    
    plt.title(f'Vector Potential A in the xy-plane at z = {z_fixed}', fontsize=35)
    plt.savefig('vector-potential-xy-plane.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


# Plot quiver diagram in the zy-plane (x=0)
def plot_zy_plane():
    fig, ax = plt.subplots(figsize=(20, 16), dpi=600)

    ax.quiver(Y_zy, Z_zy, Ay_zy, Az_zy)  # Quiver plot of the vector potential in the zy-plane

    ax.set_xlim((ymin, ymax))  # Set the xlim to ymin, ymax
    ax.set_ylim((zmin, zmax))  # Set the ylim to zmin, zmax

    plt.xlabel('Y axis', fontsize=30)
    plt.ylabel('Z axis', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    
    plt.title(f'Vector Potential A in the zy-plane (x=0)', fontsize=35)
    plt.savefig('vector-potential-zy-plane.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# Função para calcular o vetor potencial A em função da distância radial r
def plot_A_vs_r():
    r_values = np.linspace(0.01, 2.0, gs)  # Valores de r variando de perto do eixo até 2 mm
    A_values = np.zeros(gs)  # Array para armazenar os valores de A para cada r
    
    # Calcula o vetor potencial A para cada valor de r
    for i, r in enumerate(r_values):
        pos_r = np.array([r, 0, z_fixed])  # Ponto de observação a distância r do eixo
        A_r = find_A(pos_r, theta, R, N, wr)  # Calcular o vetor potencial no ponto
        A_values[i] = LA.norm(A_r)  # Armazena o valor da norma de A (magnitude)
    
    # Plotar A em função de r
    plt.figure(figsize=(10, 8))
    plt.plot(r_values, A_values, label=r'$|A|$ vs r', color='b')
    plt.xlabel('Distância radial r (mm)', fontsize=20)
    plt.ylabel('Potencial Vetorial |A| (T.mm)', fontsize=20)
    plt.title('Potencial Vetorial A em função da distância radial r', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.savefig('A_vs_r.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':
    # Initialize theta
    for i in range(0, theta.size):
        theta[i] = i * 2 * np.pi / N

    # Compute the vector potential in both planes
    norms_xy, norms_zy = find_field()

    # Plotar A em função da distância radial r
    plot_A_vs_r()

    # Plot the vector potential field for both xy-plane and zy-plane
    plot_xy_plane()
    plot_zy_plane()
