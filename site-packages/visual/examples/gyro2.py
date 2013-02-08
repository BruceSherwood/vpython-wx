from visual import *
#from visual.graph import *

# Gyroscope sitting on a pedestal

# The analysis is in terms of Lagrangian mechanics.
# The Lagrangian variables are polar angle theta,
# azimuthal angle phi, and spin angle psi.

# Bruce Sherwood

scene.width=800
scene.height=800
scene.title='Nutating Gyroscope'

Lshaft = 1. # length of gyroscope shaft
r = Lshaft/2. # distance from support to center of mass
Rshaft = 0.03 # radius of gyroscope shaft
M = 1. # mass of gyroscope (massless shaft)
Rrotor = 0.4 # radius of gyroscope rotor
Drotor = 0.1 # thickness of gyroscope rotor
I3 = 0.5*M*Rrotor**2 # moment of inertia of gyroscope about its own axis
I1 = M*r**2 + .5*I3 # moment of inertia about a line through the support, perpendicular to the axis
hpedestal = Lshaft # height of pedestal
wpedestal = 0.1 # width of pedestal
tbase = 0.05 # thickness of base
wbase = 3*wpedestal # width of base
g = 9.8
Fgrav = vector(0,-M*g,0)
top = vector(0,0,0) # top of pedestal

theta = 0.3*pi # initial polar angle of shaft (from vertical)
thetadot = 0 # initial rate of change of polar angle
psi = 0 # initial spin angle
psidot = 30 # initial rate of change of spin angle (spin ang. velocity)
phi = -pi/2 # initial azimuthal angle
phidot = 0 # initial rate of change of azimuthal angle
if False: # Set to True if you want pure precession, without nutation
    a = (1-I3/I1)*sin(theta)*cos(theta)
    b = -(I3/I1)*psidot*sin(theta)
    c = M*g*r*sin(theta)/I1
    phidot = (-b+sqrt(b**2-4*a*c))/(2*a)

pedestal = box(pos=top-vector(0,hpedestal/2.,0),
                 height=hpedestal, length=wpedestal, width=wpedestal,
                 color=(0.4,0.4,0.5))
base = box(pos=top-vector(0,hpedestal+tbase/2.,0),
                 height=tbase, length=wbase, width=wbase,
                 color=pedestal.color)

gyro=frame(axis=(sin(theta)*sin(phi),cos(theta),sin(theta)*cos(phi)))
shaft = cylinder(axis=(Lshaft,0,0), radius=Rshaft, color=color.green,
                 material=materials.rough, frame=gyro)
rotor = cylinder(pos=(Lshaft/2 - Drotor/2, 0, 0),
                 axis=(Drotor, 0, 0), radius=Rrotor, color=color.gray(0.7),
                 material=materials.rough, frame=gyro)

tip = sphere(pos=gyro.pos + gyro.axis * Lshaft, color=color.yellow,
               radius=0.001*shaft.radius, make_trail=True,
               interval=5, retain=250)
tip.trail_object.radius = 0.2*shaft.radius

scene.autoscale = 0

dt = 0.0001
t = 0.
Nsteps = 20 # number of calculational steps between graphics updates

while True:
    rate(100)
    for step in range(Nsteps): # multiple calculation steps for accuracy
        # Calculate accelerations of the Lagrangian coordinates:
        atheta = sin(theta)*cos(theta)*phidot**2+(
            M*g*r*sin(theta)-I3*(psidot+phidot*cos(theta))*phidot*sin(theta))/I1
        aphi = (I3/I1)*(psidot+phidot*cos(theta))*thetadot/sin(theta)-2*cos(theta)*thetadot*phidot/sin(theta)
        apsi = phidot*thetadot*sin(theta)-aphi*cos(theta)
        # Update velocities of the Lagrangian coordinates:
        thetadot += atheta*dt
        phidot += aphi*dt
        psidot += apsi*dt
        # Update Lagrangian coordinates:
        theta += thetadot*dt
        phi += phidot*dt
        psi += psidot*dt

    gyro.axis = vector(sin(theta)*sin(phi),cos(theta),sin(theta)*cos(phi))
    # Display approximate rotation of rotor and shaft:
    gyro.rotate(angle=psidot*dt*Nsteps)
    tip.pos = gyro.pos + gyro.axis * Lshaft
    t = t+dt*Nsteps

## If you want to check conservation of energy, you could plot kinetic and potential energies, and their sum:
##    K = .5*I1*(thetadot**2+(phidot*sin(theta))**2)+.5*I3*(psidot+phidot*cos(theta))**2
##    U = M*g*r*cos(theta)
    
