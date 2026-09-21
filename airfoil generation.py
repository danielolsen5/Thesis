"""
Generates airfoil and applies hindged flap
"""
import numpy as np
import matplotlib.pyplot as plt

# Tool Toggles

Print = True
Plot = True
Debug = False
Output = False

"""
Global variables 
"""
# Airfoil type 
    # 1: Diamond
    # 2: Wedge
    # 3: NACA
    # 4: Joukowski
Type_a = 2

# Flap Type
    # 1: leading Edge 
    # 2: Trailing Edge
    # 3: No Flap
type_f = 3

c_f = .2                # Flap Chord Percent 
phi =  np.radians(10)   # Flap Deflection Angle
n = 200                 # Number of Points

""" 
Generate Base Airfoil
"""

if Type_a == 1:
    # Diamond Generation    
    t = .1 # thickness
    e = np.radians(10) # Wedge Angle
    
    # Cos Clustering
    beta = np.linspace(0, np.pi, n//2)
    x = .5 * (1 - np.cos(beta)) #chrod length of 1
    
    y_u = np.where(x <= 0.5, x * np.tan(e/2), (1 - x) * np.tan(e/2))
    y_l = -1 * y_u
    x_u = x
    x_l = x[::-1]
       
    x = np.concatenate((x_l, x_u))
    y = np.concatenate((y_l, y_u))
    
elif Type_a == 2:
    # Wedge Generation
    
    e = np.radians(10) #Wedge Angle
    
    # Cos Clustering
    beta = np.linspace(0, np.pi, n//2)
    x = .5 * (1 - np.cos(beta)) #chrod length of 1
    y_u = np.zeros(n//2)
    y_l = np.zeros_like(y_u)
    
    y_u = x * np.tan(e/2)
    y_l = y_u * -1 
    y_l = y_l[::-1]
    x_u = x
    x_l = x[::-1]
    
    x = np.concatenate((x_l, x_u))
    y = np.concatenate((y_l, y_u))
    
elif Type_a == 3:
    # NACA Generation
            NACA = input("Enter NACA 4 digit code: ")
            M = int(NACA[0]) / 100 # camber
            P = int(NACA[1]) / 10 # max camber position
            T = int(NACA[2:]) / 100 # thickness
            
            # closed airfoil coefficients 
            a_0 = .298
            a_1 = -.132
            a_2 = -.3286
            a_3 = .2441
            a_4 = -.0815
            
            # cos spacing
            beta = np.linspace(0, np.pi, n)
            x = .5 * (1 - np.cos(beta)) #chrod length of 1

            #Thickness distribution
            y_t = (T/.2) * ((a_0 * x**.5) + (a_1 * x) + (a_2 * x**2) + (a_3 * x**3) + (a_4 * x**4))

            #camber line
            y_c = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
            for i, xi in enumerate(x):
                if xi < P:
                    y_c[i] = M / (P**2) * (2*P*xi - xi**2)
                    dyc_dx[i] = 2 * M / (P**2) * (P - xi)
                else:
                    y_c[i] = M / (1-P)**2 * ((1 - 2*P) + 2*P*xi - xi**2)
                    dyc_dx[i] = 2 * M / (P**2) * (P - xi)
                
            theta = np.arctan(dyc_dx)

            y_u = y_c + y_t*np.cos(theta)
            y_l = y_c - y_t*np.cos(theta)
            x_u = x - y_t*np.sin(theta)
            x_l = x + y_t*np.sin(theta)

            x = np.concatenate((np.flip(x_u), x_l[1:]))
            y = np.concatenate((np.flip(y_u), y_l[1:]))
            
            if Debug:
                print("length of x", len(x))
                print("Length of y", len(y))
        
elif Type_a == 4:
    
    # Joukowski Generation    
    R = 1.0  # Radius of the circular cylinder in the zeta plane
    xi0 = -0.09 # Real center "x"  drives thickness
    eta0 = 0.1 # Imaginary center "y" drives camber/symetry 
    eps = R - np.sqrt(R**2 - eta0**2) - xi0 # Eccentricity parameter  drives trailing edge sharpness
    V_inf = 10          # freestream velocity 
    a = np.radians(0)

    gamma_k = 4 * np.pi * V_inf * (np.sqrt(R**2 - eta0**2)*np.sin(a) + eta0*np.cos(a)) #Kutta circulation Eq 45

    if abs(gamma_k) <= (4 * np.pi * V_inf * R):
        theta_stag_aft = a - np.arcsin(gamma_k / (4 * np.pi * V_inf * R))
        theta_stag_fwd = np.pi - theta_stag_aft + 2 * a
    else:
        theta_stag_fwd = a - np.arcsin(gamma_k / (4 * np.pi * V_inf * R))
        theta_stag_aft = np.pi - theta_stag_fwd + 2 * a

    xi_stag_aft = R * np.exp(1j * theta_stag_aft) + xi0
    xi_stag_fwd = R * np.exp(1j * theta_stag_fwd) + xi0

    def transform_joukowski_cylinder(n, R, xi0, eta0, eps, theta_stag_aft):
        
        # Define the center of the circular cylinder in the ζ-plane (Eq. 19)
        zeta_0 = xi0 + 1j * eta0
        
        # Create an array of angles (θ) to parameterize the circle
        theta = np.linspace(0, 2*np.pi, n)
        theta = theta + theta_stag_aft

        # Define the surface of the circular cylinder in the ζ-plane (Eq. 20)
        zeta_surface = R * np.exp(1j * theta) + zeta_0
        
        # Apply the Joukowski transformation to the surface points (Eq. 18)
        z_surface = zeta_surface + (R - eps)**2 / zeta_surface
        
        return zeta_surface, z_surface

    # Perform the transformation
    zeta_surface, z_surface = transform_joukowski_cylinder(n, R, xi0, eta0, eps, theta_stag_aft)

    # Separate real and imaginary parts for analysis or plotting
    z_real_0 = z_surface.real
    z_imag_0 = z_surface.imag

    # Shift leading edge to origin
    x_max = np.max(z_real_0) #maximum "x" value
    x_min = np.min(z_real_0) #minimum "x" vlaue

    offset = x_min

    if offset < 0:
        z_real_t =  z_real_0 + abs(offset)
    else:
        z_real_t = z_real_0 - abs(offset)

    # scale chord length to 1
    chord_scale = 1 / np.max(z_real_t)

    z_real_f = chord_scale * z_real_t
    z_imag_f = chord_scale * z_imag_0

    # flip vlaues to follow clockwise convention 
    x = z_real_f[::-1]
    y = z_imag_f[::-1]


    if Debug:
        print("X min ", x_min)
        print("X max", x_max)
        print("chord scale: ", chord_scale)
        print("Original Coordinates", zeta_surface)
        print("Transformed Real Components (z_real):\n", x)
        print("\nTransformed Imaginary Components (z_imag_0):\n", y)

"""
Apply Flap Deflection
"""    

if type_f == 1:
# Leading Edge Flap
    
    # Translate upper surface
    y_u = np.where(x_u <= c_f, y_u - ((c_f - x_u) * np.tan(phi)), y_u)
    
    # Translate lower surface 
    y_l = np.where(x_l <= c_f, y_l - ((c_f - x_l) * np.tan(phi)), y_l)   
    
    y = np.concatenate((y_l, y_u))
    
elif type_f == 2:
# Trailing Edge Flap
    x_hinge = 1 - c_f
    
    # Translate upper surface
    y_u = np.where(x_u <= x_hinge, y_u, y_u - ((x_u - x_hinge) * np.tan(phi)))
    
    # Translate lower surface 
    y_l = np.where(x_l <= x_hinge, y_l, y_l - ((x_l - x_hinge) * np.tan(phi)))   
    
    y = np.concatenate((y_l, y_u))
else:
    x = x
    y = y
    
    
"""
Printouts 
"""

if Print:

    if Type_a == 1:
        # Diamond Generation    
        print("Diamond Airfoil" )
        print(f"Thickness: {t*100:.1f}%")
        print(f"Wedge Angle {np.degrees(e):.1f} deg")
    elif Type_a == 2:
        # Wedge Generation
        z=2
    elif Type_a == 3:
        # NACA Generation
    
        z=2
    elif Type_a == 4:
        # Joukowski Generation    
        z=4
        
    print(f"Flap Deflection Angle {np.degrees(phi):.1f} deg, Percent Chord {c_f*100:.1f}%")
    
    """
    Plotting
    """
if Plot:
        
        # Plotting the airfoil
        plt.figure(figsize=(10, 2))
        plt.scatter(x, y, s=2)
        if Type_a == 1:
            plt.title('Diamond Airfoil')
        elif Type_a == 2:
            plt.title('Wedge Airfoil')
        elif Type_a == 3:
            plt.title(f'NACA {NACA} Airfoil')
        elif Type_a == 4:
            plt.title('Joukowski Airfoil')
            
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
if Output:
    #output the airfoil
    np.savetxt(f"airfoil.txt", np.column_stack((x, y)), fmt="%.6f")