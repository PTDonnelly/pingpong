"""
Basic kinematics model for computing ping pong serves. Forms a training set
for neural network practice. This is a rough and UNoptimised code, useful for
plotting individual trajectories.
PTD 06/07/21
"""
#%%
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

#############################
##### Kinematics inputs #####
#############################

# Physical constants
g = 9.80665                                          # Accleration due to gravity, m/s^2

# Physical parameters
table_length =  1.372                                # metres
# table_width =   0.762                              # metres
net_height =    0.1525                               # metres
# ball_radius =    0.02                              # metres
# ball_mass =     0.0027                             # kilograms

paddle_range  = [0.3]
# paddle_range  = np.arange(0.1, 1, 0.1)               # Serve heights, (m)
initial_v     = np.arange(1, 10, 1)                 # Initial velocities (m/s)
initial_theta = np.arange(-90,90, 1)                # Trajectory angles (degrees)
time_range    = np.arange(0, 5, 0.001)               # Time domain (s), set as 'continous'

# for testing
# paddle_range  = np.arange(0.1, 1, 0.01)            # Serve heights, (m)
# initial_v     = np.arange(1, 10, 0.1)              # Initial velocities (m/s)
# initial_theta = np.arange(-90, 91, 3)              # Trajectory angles (degrees)
# time_range    = np.arange(0, 5, 0.01)              # Time domain (s), set as 'continous'


n_p           = len(paddle_range)
n_vel         = len(initial_v)
n_theta       = len(initial_theta)
n_time        = len(time_range)
n_hop         = 2                                   # no.of "hops" (parabolas)
print(n_p, n_vel, n_theta, n_time, n_hop, '=', n_p*n_vel*n_theta,'paths')


#######################
##### Data arrays #####
#######################
path_x = np.zeros((n_theta, n_time, n_hop))         # x-values of paths
path_y = np.zeros((n_theta, n_time, n_hop))         # y-values of paths
path_s = np.zeros((n_theta, n_hop))                 # range values of paths
path_h = np.zeros((n_theta, n_hop))                 # apse values of paths
output = np.zeros((n_p*n_vel*n_theta, 5))           # for storing the training set
next_v = np.zeros(n_theta)                          # velocities of subsequent hops
next_theta = np.zeros(n_theta)                      # angles of reflection of subsequent hops

# def scattering(v, theta, mode):
    
# @profile

def serve_plot():

    start_time = time.time()
    for ip, p in enumerate(paddle_range):
        current_time1 = time.time() 
        
        idx1 = ip*n_vel*n_theta
        idx2 = (ip+1)*n_vel*n_theta
        output[idx1:idx2, 0] = p                                                    # store paddle_height
        
        # Boundary conditions
        net_shadow = (net_height * (0.5*table_length)) / (p + net_height)
        s_min = (0.5 * table_length) + net_shadow
        s_max = table_length
        
        plt.figure(figsize=(8,8), dpi=300)
        plt.suptitle('Trajectories: '+str(np.round(p, 2))+' m')
        cmap = plt.get_cmap('jet')
    
        for iv, v in enumerate(initial_v):
            idx1 = (ip*n_vel*n_theta) + (iv*n_theta)
            idx2 = (ip*n_vel*n_theta) + ((iv+1)*n_theta)
            output[idx1:idx2, 1] = v                                                # store initial_velocity
            
            ax1 = plt.subplot2grid((3, 3), (iv // 3, iv % 3))
            ax1.set_title('v = '+str(np.round(v, 2))+' m/s', fontsize=10, pad=2)
            ax1.set_xlabel('Range (m)', fontsize=8, labelpad=2.5)
            ax1.set_xlim([0, table_length])
            ax1.set_xticks(np.arange(0, table_length+0.01, 0.2))
            ax1.set_ylim([-p, 2])
            ax1.set_ylabel('Height (m)', fontsize=8, labelpad=2.5)
            ax1.tick_params(axis='both', length=2.5, pad=2, labelsize=7)
            
            for ihop in range(n_hop):
                if ihop == 0:
                   
                    # Calculate trajectory for every theta
                    for itheta, theta in enumerate(initial_theta):
                        col = cmap(itheta/n_theta)    
                        
                        idx1 = (ip*n_vel*n_theta) + (iv*n_theta) + itheta
                        output[idx1, 2] = theta                                # store initial_theta
                        
                        # Calculate position at each time step
                        for it, t in enumerate(time_range):
                            x = ((v*t) * cos(radians(theta)))
                            y = ((v*t) * sin(radians(theta)) - ((0.5*g) * (t**2)))
                            path_x[itheta, it, ihop] = x
                            path_y[itheta, it, ihop] = y
                            
                        # Reflect path off the table
                        tmp = [a for a, b in enumerate(path_y[itheta, :, ihop]) if b < -p] 
                        for a in sorted(tmp, reverse = True):
                            path_x[itheta, a, ihop] = 'NaN'
                            path_y[itheta, a, ihop] = 'NaN'
                        path_s[itheta, ihop] = np.nanmax(path_x[itheta, :, ihop])
                        path_h[itheta, ihop] = np.nanmin(path_y[itheta, :, ihop])
                        
                        # Calculate angles of reflection for next hop
                        if theta < 0:                                                     # origin (downward shots) 
                            x_zero = 0
                        if theta > 0:                                                      # zero-crossing (upward shots)
                            keep = np.where(path_x[itheta, :, ihop] > 0)[0]
                            f = interpolate.interp1d(path_y[itheta, keep, ihop],
                                                     path_x[itheta, keep, ihop],
                                                     fill_value="extrapolate")
                            x_zero = f(0)
                        dx = path_s[itheta, ihop] - x_zero
                        dtheta = atan(p/dx)                                                
                        if theta == 0:                                                     # origin (horizontal shots) 
                            keep = np.where(path_x[itheta, :, ihop] == path_s[itheta, ihop])[0]
                            dt = time_range[keep]
                            dtheta = atan((g*dt)/v) 
                        next_theta[itheta] = radians(abs(theta)+dtheta)
                        # print(v, theta, dtheta)
                        
                        # Plot initial trajectories
                        ax1.plot(path_x[itheta, :, ihop], path_y[itheta, :, ihop], color=col, alpha=0.1)
                    
                        # Calculate velocities for next hop
                        next_v[itheta] = 0.94*v                                              # elastic scattering, i.e. v_i = v_0  
                    
                if ihop == 1:
                    
                    # Calculate trajectory for every theta
                    for itheta, theta in enumerate(next_theta):
                        v = next_v[itheta]    
                        
                        # Calculate position at each time step
                        for it, t in enumerate(time_range):
                            x = ((v*t) * cos(theta))
                            y = ((v*t) * sin(theta) - ((0.5*g) * (t**2)))
                            path_x[itheta, it, ihop] = x + path_s[itheta, ihop-1]
                            path_y[itheta, it, ihop] = y + path_h[itheta, ihop-1]
                            
                        # Reflect path off table
                        tmp = [a for a, b in enumerate(path_y[itheta, :, ihop]) if b < -p] 
                        for a in sorted(tmp, reverse = True):
                            path_x[itheta, a, ihop] = 'NaN'
                            path_y[itheta, a, ihop] = 'NaN'
                        path_s[itheta, ihop] = np.nanmax(path_x[itheta, :, ihop])
                        path_h[itheta, ihop] = np.nanmin(path_y[itheta, :, ihop])
            
            # Reject invalid paths (illegal serves) and store logistic values
            for itheta in range(n_theta):
                idx1 = (ip*n_vel*n_theta) + (iv*n_theta) + itheta
                output[idx1, 3] = 1                                                 # store serve as valid (1)
                
                f = interpolate.interp1d(path_x[itheta, :, ihop],
                                         path_y[itheta, :, ihop],
                                         fill_value="extrapolate")
                y_at_net = f(0.5*s_max)
                if (path_s[itheta, 0] > 0.5*table_length or                         # Path does touch server's area
                    y_at_net < (net_height-p) or
                    path_s[itheta, -1] > s_max or                                   # Path ends beyond the table (ball is out)
                    path_s[itheta, -1] < s_min):                                    # Path ends before net shadow (ball does not clear the net)
                        path_x[itheta, :, :] = 'NaN'
                        path_y[itheta, :, :] = 'NaN'
                        output[idx1, 3] = 0                                         # override element if invalid (0)
                        
                # Plot trajectories
                col = cmap(itheta/n_theta)  
                ax1.plot(path_x[itheta, :, 0], path_y[itheta, :, 0], color=col)
                ax1.plot(path_x[itheta, :, 1], path_y[itheta, :, 1], color=col)
            ax1.plot([s_min, s_min], [-p, 2], color='black', lw=0.5, ls='--')
            ax1.plot([0.5*table_length]*2, [-p, 2], color='black', lw=0.5)
            
            # Clean up axes
            if iv % 3 != 0:
                ax1.set_yticks([])
                ax1.set_ylabel('')
            
            if iv // 3 < 2:
                ax1.set_xticks([])
                ax1.set_xlabel('')
            
        plt.subplots_adjust(top=0.9, hspace=0.1, wspace=0.1)
        # plt.savefig('basic_serve_'+str(ip)+'.png', dpi=300)
        plt.show()
        
        current_time2 = time.time() 
        elapsed_time = current_time2 - current_time1
        
        # Calcuate the percentage of legal serves
        n_runs = n_vel*n_theta
        idx1 = ip*n_vel*n_theta
        idx2 = (ip+1)*n_vel*n_theta
        legal = (output[idx1:idx2, 3] == 1).sum()
        print(f'{legal}/{n_runs} runs ({np.round(100*(legal/n_runs), 3)} %) in {np.round(elapsed_time, 3)} s')
        
        idx1 = ip*n_vel*n_theta
        idx2 = (ip+1)*n_vel*n_theta
        output[idx1:idx2, 4] = 100*(legal/n_runs)
        
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print('Total time: ', np.round(elapsed_time, 3), ' s')
    
serve_plot()

