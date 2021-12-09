import numpy as np
from dolfin import *

# Model parameters
epsilon = 0.01
dt = 1.0e-05  # time step
SaveStep = 1000
# theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
    #     random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        cord_x = np.array([np.pi/2, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, np.pi, 3*np.pi/2])
        cord_y = np.array([np.pi/2, 3*np.pi/4, 5*np.pi/4, np.pi/4, np.pi/4, np.pi, 3*np.pi/2])
        cord_r = np.array([np.pi/10, np.pi/8, np.pi/6, np.pi/6, np.pi/5, np.pi/4, np.pi/4])
        cord_x = cord_x/(2*np.pi)
        cord_y = cord_y/(2*np.pi)
        cord_r = cord_r/(2*np.pi)

        values[0] = 0.0
        for i in range(7):
            values[0] = values[0] + 0.5*(1.0-np.tanh((np.sqrt((x[0]-cord_x[i])**2+(x[1]-cord_y[i])**2)-cord_r[i])/0.02))

# Class for interfacing with the Newton solver
class AC(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

# Create mesh and build function space
mesh = UnitSquareMesh.create(128, 128, CellType.Type.quadrilateral)# A unit square mesh with 129 (= 128 + 1) vertices in each direction is created
V = FunctionSpace(mesh, 'CG', 1)

#Define trial and test functions
du = TrialFunction(V)
v = TestFunction(V)

#Define function
u = Function(V)  # current solution
u0 = Function(V)  # solution from previous converged step

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)

L = u*v*dx - u0*v*dx + epsilon**2*dt*dot(grad(u),grad(v))*dx-dt*(u**3-u)*v*dx

a = derivative(L,u,du)

# Create nonlinear problem and Newton solver
problem = AC(a, L)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "gmres"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-12

def total_free_energy(epsilon,u):
    return assemble(epsilon**2/2.0*dot(grad(u),grad(u))*dx + 1/4.0*(u**2-1)**2*dx)

def total_volume(u):
    return assemble(u*dx)

# Output file
file_u = File("AC.pvd", "compressed")

# Step in time
t = 0
time = 0.02
T = time/dt
energy_volume=[]
while (t <= T):
    if t%SaveStep==0:
        file_u << (u, t)

    E_total = total_free_energy(epsilon,u)
    V_total = total_volume(u)
    energy_volume.append([t*dt,E_total,V_total])

    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())
    t += 1

np.savetxt('AC_energy_volume.csv',np.array(energy_volume), fmt='%1.10f', header = 'time,total_free_energy,total_volume',delimiter=',',comments='')



