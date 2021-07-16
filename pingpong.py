"""
Basic kinematics model for computing ping pong serves. Forms a training set
for neural network practice.
PTD 06/07/21
PTD 12/07/21: Re-written in terms of class attributes and methods
"""
import numpy as np
from math import sin, cos, atan, radians
import matplotlib.pyplot as plt
import time
import cProfile, pstats, io
from scipy import interpolate
       
def profile(fnc):
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner 
     
# @profile   
class Model():
    
    def __init__(self):
        """Parameters used to initialise the Model class. This includes
        physical constants, geometric contraints of the system and the 
        output arrays."""
        
        ##### Physical constants
        self.g             = 9.80665                                            # accleration due to gravity (m/s^2)
        self.Earth_mass    = 5.972e24                                           # mass of the Earth (kg)
        
        ##### Physical parameters
        self.table_length  = 1.372                                              # (m)
        # self.table_width = 0.762                                              # (m)
        self.net_height    = 0.1525                                             # (m)
        # self.ball_radius = 0.02                                               # (m)
        self.ball_mass     = 2.7e-3                                             # (kg)
        self.CR            = 0.92                                               # coefficient of restitution
        self.paddle_range  = np.arange(0.1, 1, 0.1)                             # serve heights, (m)
        self.initial_v     = np.arange(1, 10, 0.3)                              # initial velocities (m/s)
        self.initial_theta = np.arange(-90, 90, 6)                              # trajectory angles (degrees)
        self.times         = np.arange(0, 5, 0.01)                              # time domain (s), set as 'continous'
        
        ##### Data arrays
        self.Np            = len(self.paddle_range)                             # no. of serve heights
        self.Nv            = len(self.initial_v)                                # no. of serve speeds
        self.Na            = len(self.initial_theta)                            # no. of serve angles
        self.Nt            = len(self.times)                                    # no. of time steps
        self.Nh            = 2                                                  # no.of "hops" (parabolas)
        self.path_x        = np.zeros((self.Na, self.Nt, self.Nh))              # x-values of paths
        self.path_y        = np.zeros((self.Na, self.Nt, self.Nh))              # y-values of paths
        self.path_s        = np.zeros((self.Na, self.Nh))                       # range values of paths
        self.path_h        = np.zeros((self.Na, self.Nh))                       # apse values of paths
        self.output        = np.zeros((self.Np*self.Nv*self.Na, 5))             # for storing the training set
        self.next_v        = np.zeros(self.Nt)                                  # velocities of subsequent hops
        self.next_theta    = np.zeros(self.Na)                                  # angles of reflection of subsequent hops
        print(self.Np, self.Nv, self.Na,'=', self.Np*self.Nv*self.Na,'paths')
        print(self.Nt, self.Nh, '=', self.Nt*self.Nh, 'points per path')
        print('-----')
    
    def serve(self):
        """Takes in boundary constraints and kinematic inputs to compute 
        parabolas and store logistic value of serve (0) illegal or (1) legal.
        Training set (input parameters and logistic value) is then written
        to a .csv file."""
        
        start_time = time.time()
        for ip, p in enumerate(self.paddle_range):                  
            current_time1 = time.time() 
            # Store serve height
            self.idx_output('p', [ip], p)
            # Get bounds
            net_shadow, s_min, s_max = self.bounds(p)
            
            for iv, v in enumerate(self.initial_v):                            
                # Store initial_velocity    
                self.idx_output('v', [ip, iv], v)
                
                # Calculate trajectories
                for ihop in range(self.Nh):
                    if ihop == 0:
                        for itheta, theta in enumerate(self.initial_theta):    
                            # Store initial_theta    
                            self.idx_output('theta', [ip, iv, itheta], theta)
                            self.parabola(p, v, theta, itheta, ihop)
                    if ihop == 1:
                        for itheta, theta in enumerate(self.next_theta):
                            # v = self.next_v[itheta]    
                            # theta = self.next_theta[itheta]    
                            self.parabola(p, v, theta, itheta, ihop)
                            
                # Store logistic values and reject illegal serves
                for itheta in range(self.Na):
                    # Store serve as legal (1)   
                    self.idx_output('result', [ip, iv, itheta], 1)
                    
                    f = interpolate.interp1d(self.path_x[itheta, :, ihop],
                                             self.path_y[itheta, :, ihop],
                                             fill_value="extrapolate")
                    y_at_net = f(0.5*s_max)
                    if (self.path_s[itheta, 0] > 0.5*s_max or                   # Path does not first touch server's area
                        y_at_net < (self.net_height-p) or                       # Path does not clear the net
                        self.path_s[itheta, 1] < s_min or                       # Path ends before net shadow
                        self.path_s[itheta, 1] > s_max):                        # Path ends beyond the table
                            self.path_x[itheta, :, :] = 'NaN'
                            self.path_y[itheta, :, :] = 'NaN'
                            # Override serve if illegal (0)   
                            self.idx_output('result', [ip, iv, itheta], 0)
                    
            current_time2 = time.time() 
            elapsed_time = current_time2 - current_time1
            
            # Calcuate the percentage of legal serves
            n_runs = self.Nv*self.Na
            idx1 = ip*self.Nv*self.Na
            idx2 = (ip+1)*self.Nv*self.Na
            legal = (self.output[idx1:idx2, 3] == 1).sum()
            print(f'{legal}/{n_runs} runs ({np.round(100*(legal/n_runs), 3)}%) in {np.round(elapsed_time, 3)} s')
            self.output[idx1:idx2, 4] = 100*(legal/n_runs)
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print('Total time: ', np.round(elapsed_time, 3), ' s')
        print(' ')
        
        # Save output array training set for neurall network
        # np.savetxt("training_set.csv", output, delimiter=",")
        
        # Plot scatter of results in parameter space
        self.plot_params()
        # self.plot_params('HvsX')
        

    def parabola(self, p, v, theta, itheta, ihop):
        """Calculate parabola for any hop based on velocity and 
        angle inputs."""
        
        # Calculate position at each time step
        for it, t in enumerate(self.times):
            x = ( (v * t) * cos(radians(theta)) )
            y = ( (v * t) * sin(radians(theta)) - ( (0.5 * self.g) * (t**2) ) )
            if ihop == 0:
                self.path_x[itheta, it, ihop] = x
                self.path_y[itheta, it, ihop] = y
            if ihop == 1:
                self.path_x[itheta, it, ihop] = x + self.path_s[itheta, ihop-1]
                self.path_y[itheta, it, ihop] = y + self.path_h[itheta, ihop-1]
            
        # Reflect path off the table
        tmp = [a for a, b in enumerate(self.path_y[itheta, :, ihop]) if b < -p]
        for a in sorted(tmp, reverse = True):
            self.path_x[itheta, a, ihop] = 'NaN'
            self.path_y[itheta, a, ihop] = 'NaN'
        self.path_s[itheta, ihop] = np.nanmax(self.path_x[itheta, :, ihop])
        self.path_h[itheta, ihop] = np.nanmin(self.path_y[itheta, :, ihop])
        
        if ihop < self.Nh-1:
            # Calculate angles of reflection for next hop (theta + dtheta)
            # if theta < 0:                                                      # origin (downward shots) 
            x_zero = 0
            if theta > 0:                                                      # zero-crossing (upward shots)
                keep = np.where(self.path_x[itheta, :, ihop] > 0)[0]
                f = interpolate.interp1d(self.path_y[itheta, keep, ihop],
                                         self.path_x[itheta, keep, ihop],
                                         fill_value="extrapolate")
                x_zero = f(0)
            dx = self.path_s[itheta, ihop] - x_zero
            dtheta = atan(p/dx)                                                
            if theta == 0:                                                     # origin (horizontal shots) 
                keep = np.where(self.path_x[itheta, :, ihop] == self.path_s[itheta, ihop])[0]
                dt = self.times[keep]
                dtheta = atan((self.g*dt)/v) 
            self.next_theta[itheta] = abs(theta)+dtheta
            
            # Calculate velocities for next hop
            m1, m2, u1, u2, CR = self.ball_mass, self.Earth_mass, v, 0, self.CR
            self.next_v[itheta] = abs(((CR * m2 * (u2 - u1)) + (m1 * u1) \
                                   + (m2 * u2)) / ( m1 + m2 ))                 

                
    def bounds(self, paddle_height):
        """Calculate boundary values based on the geometric (must clear
        the net and land in the opponents area) and rule (must first enter 
        the server's area) constraints of a 'legal' serve."""
        
        # Boundary values
        net_shadow = (self.net_height * (0.5*self.table_length)) / \
                     (paddle_height + self.net_height)
        s_min = (0.5 * self.table_length) + net_shadow
        s_max = self.table_length
        return (net_shadow, s_min, s_max)
    
    
    def idx_output(self, loop, i, value):
        """Output parameter values and their logistic value (legality
        of serve) to an array for saving."""
                                                          
        if loop == 'p':
            ip = i[0]
            idx1 = ip*self.Nv*self.Na
            idx2 = (ip+1)*self.Nv*self.Na
            self.output[idx1:idx2, 0] = value
        if loop == 'v':
            ip, iv = i[0], i[1]
            idx1 = (ip*self.Nv*self.Na) + (iv*self.Na)
            idx2 = (ip*self.Nv*self.Na) + ((iv+1)*self.Na)
            self.output[idx1:idx2, 1] = value
        if loop == 'theta':
            ip, iv, itheta = i[0], i[1], i[2]
            idx1 = (ip*self.Nv*self.Na) + (iv*self.Na) + itheta
            self.output[idx1, 2] = value   
        if loop == 'result':
            ip, iv, itheta = i[0], i[1], i[2]
            idx1 = (ip*self.Nv*self.Na) + (iv*self.Na) + itheta
            self.output[idx1, 3] = value    
            
            
    def plot_params(self):
        """Plot the parameter space:
                Display as a scatter plot all serves for each velocity-angle
                point as a function of serve height. This gives a measure of the
                proportion of legal serves in the space and will be compared 
                between different physical constraints."""
        
        plt.figure(figsize=(7,7), dpi=300)
        plt.suptitle('Comparison: Velocity vs. Serve Angle')
        xlim = [self.initial_theta[0], self.initial_theta[-1]]
        xticks= np.arange(xlim[0], xlim[1]+1, 30)
        ylim = [self.initial_v[0], self.initial_v[-1]]
        yticks = np.arange(ylim[0], ylim[1], 1)
        for ip, p in enumerate(self.paddle_range):
            ax1 = plt.subplot2grid((3, 3), (ip // 3, ip % 3))
            ax1.set_title('p = '+str(np.round(p, 2))+' m', fontsize=10, pad=2)
            ax1.set_xlabel(r'Initial $\theta$ (deg)', fontsize=8, labelpad=2.5)
            ax1.set_xlim(xlim)
            ax1.set_xticks(xticks)
            ax1.set_ylabel('Initial velocity (m/s)', fontsize=8, labelpad=2.5)
            ax1.set_ylim(ylim)
            ax1.set_yticks(yticks)
            ax1.tick_params(axis='both', length=2.5, pad=2, labelsize=7)
            
            find = np.where(self.output[:, 0] == p)
            bad = np.where(self.output[find, 3] == 0)
            good = np.where(self.output[find, 3] == 1)
            
            ax1.scatter(self.output[bad, 2], self.output[bad, 1], 
                        marker='.', color='lightgrey', s=20)
            ax1.scatter(self.output[good, 2], self.output[good, 1], 
                        marker='.', color='orangered', s=20)
            ax1.plot([0, 0], ylim, color='black', lw=0.5, ls='--')
            
            # Clean up axes
            if ip % 3 != 0:
                ax1.set_yticks([])
                ax1.set_ylabel('')
            if ip // 3 < 2:
                ax1.set_xticks([])
                ax1.set_xlabel('')
        
        # Display and close plot
        plt.subplots_adjust(top=0.9, hspace=0.1, wspace=0.1)
        # plt.savefig('compare_velocityVsangle.png', dpi=300)
        plt.show()
            
    # def plot_paths(self):
    #     """Plot trajectories:
    #             Display as a scatter plot all serves for each velocity-angle
    #             point as a function of serve height. This gives a measure of the
    #             proportion of legal serves in the space and will be compared 
    #             between different physical constraints."""
        
            
    #     for ip, p in enumerate(self.paddle_range):
    #         _, s_min, s_max = self.bounds(p)
    #         plt.figure(figsize=(8,8), dpi=300)
    #         plt.suptitle('Trajectories: '+str(np.round(p, 2))+' m')
    #         cmap = plt.get_cmap('jet')
    #         for iv, v in enumerate(self.initial_v):
    #             ax1 = plt.subplot2grid((3, 3), (iv // 3, iv % 3))
    #             ax1.set_title('v = '+str(np.round(v, 2))+' m/s', fontsize=10, pad=2)
    #             ax1.set_xlabel('Range (m)', fontsize=8, labelpad=2.5)
    #             ax1.set_xlim([0, s_max])
    #             ax1.set_xticks(np.arange(0, s_max+0.01, 0.2))
    #             ax1.set_ylim([-p, 2])
    #             ax1.set_ylabel('Height (m)', fontsize=8, labelpad=2.5)
    #             ax1.tick_params(axis='both', length=2.5, pad=2, labelsize=7)
                
    #             # Plot initial trajectories
    #             for itheta, theta in enumerate(self.initial_theta):
    #                 # col = cmap(itheta/n_theta)    
    #                 # ax1.plot(path_x[itheta, :, ihop], path_y[itheta, :, ihop], color=col, alpha=0.1)
                    
    #                 col = cmap(itheta/self.Na)  
    #                 ax1.plot(self.path_x[itheta, :, 0], self.path_y[itheta, :, 0], color=col)
    #                 ax1.plot(self.path_x[itheta, :, 1], self.path_y[itheta, :, 1], color=col)
    #             ax1.plot([s_min, s_min], [-p, 2], color='black', lw=0.5, ls='--')
    #             ax1.plot([0.5*s_max]*2, [-p, 2], color='black', lw=0.5)
                
    #             # Clean up axes
    #             if iv % 3 != 0:
    #                 ax1.set_yticks([])
    #                 ax1.set_ylabel('')
                
    #             if iv // 3 < 2:
    #                 ax1.set_xticks([])
    #                 ax1.set_xlabel('')
            
    #     plt.subplots_adjust(top=0.9, hspace=0.1, wspace=0.1)
    #     # plt.savefig('basic_serve_'+str(ip)+'.png', dpi=300)
    #     plt.show()        
s1 = Model()
s1.serve()