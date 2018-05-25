
## Q-Learning Tower of Hanoi
# Yann Adjanor, Richard Mann, Feb 2018

#standard imports
#matplotlib inline
import numpy as np
from functools import reduce
import random
import matplotlib.pyplot as plt
from time import sleep
import time
import sys
from IPython.display import display
from statsmodels.tools.eval_measures import rmse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.style.use('seaborn-whitegrid')

mypath = '/Users/ensemble/Documents/Module_Software_Agents/Coursework/'
img_path = './img_data/'
data_path = './data/'

#Global vars
nDisks = 4
Z = [0] * (nDisks) # List of the first n prime numbers
state = [0] * (3**nDisks) # List of all possible states
R=[] # R matrix
Q=[] # Q matrix
alpha0=[]
epsilon0=[]
epsilon0=[]
converged_episodes_mean0 = []
converged_episodes_std0 = []
early_stopping = False#Stop Qlearning once RMSE convergence achieved

#For dynaQ
dynaQn = 1 #0 for normal Q learning
M=[] #Dyna Q matrix

#List n first primes (greater than 1)
def prime_list(n):  
    from itertools import count, islice
    primes = (n for n in count(2) if all(n % d for d in range(2, n)))
    return list(islice(primes, 0,  n))



#Check if all the disks are present (for debugging)
def is_valid_state(nDisks, p1, p2, p3):  
    dlist = [x for l in (p1,p2,p3) for x in l]
    dlist.sort()
    return (dlist == list(range(1, nDisks+1)))


#Encode each state as a unique integer using prime factorisation: pi=list of disks on pole i, Z as above (list of the 1st n prime numbers)
def encode_state(Z, p1, p2, p3):    # o(n) complexity
    pre_code = [[Z[i-1]**(np+1) for i in pn] for np, pn in enumerate((p1,p2,p3))]
    return reduce(lambda x, y: x * y, [x for l in pre_code for x in l])


#From the state integer number, get the positions of all disks
def decode_state(nDisks, Z, sNum):    #around o(nxsqrt(n)) complexity
    p = [[] for i in range(3)]
    for i in range(nDisks):
        pn = len([i for i, e in enumerate(sNum % Z[i]**p for p in [1,2,3]) if e == 0])
        p[pn-1].append(i+1)
    return p[0], p[1], p[2]


#Define macro state: we only need to look at the value of the top disks in each pole
def macro_state(nDisks, Z, states, s): #returns a 3-tuple with the value of the top disks
    p1, p2, p3 = decode_state(nDisks, Z, states[s])
    #possible variation: no disk=largest possible disk
    #return (nDisks+1) if p1==[] else min(p1), (nDisks+1) if p2==[] 
    #else min(p2), (nDisks+1) if p3==[] else min(p3) 
    return 0 if p1==[] else min(p1), 0 if p2==[] else min(p2), 0 if p3==[] else min(p3) 


#Transition function
#define possible actions (moves) from state s
def next_allowed_states(nDisks, Z, states, s):  # complexity: same as decode_state
    A=[]
    t = macro_state(nDisks, Z, states, s)
    if t == (0,0,1):   #Goal state 
        A = [s]        #this might be too restrictive  examine potential change
    else:
        sNum = states[s]
        #print(sNum)
        for i in range(3): # examine each pole in succession
            # compare top of pole i with pole i+1, if lower or empty, move there
            if (t[i]!=0) and ((t[i] < t[(i+1)%3]) or t[(i+1)%3]==0): 
                A.append(states.index(round(sNum*(Z[t[i]-1]**((i+1)%3-i)))))
            # compare top of pole i with pole i+2, if lower or empty, move there
            if (t[i]!=0) and ((t[i] < t[(i+2)%3]) or t[(i+2)%3]==0): 
                A.append(states.index(round(sNum*(Z[t[i]-1]**((i+2)%3-i)))) ) 
    return A


#Define R matrix
def set_R_matrix(nDisks, Z, states): 
    stSpSize = 3**nDisks
    R = [[None for c in range(stSpSize)] for r in range(stSpSize)] #None value by default
    for s in range(stSpSize):
        #print(decode_state(nDisks, Z, states[s])) 
        A = next_allowed_states(nDisks, Z, states, s)
        for a in A:
            #R[s][a] = 0  #0 value for allowed states
            R[s][a] = 0  if  a !=(stSpSize-1) else 100 #0 value for allowed states
    #R[stSpSize-1][stSpSize-1] = 100  #100 for goal state
    return R


#Initialise Q matrix to zero
def init_Q_matrix(nDisks):
    stSpSize = 3**nDisks
    Q = [[0 for c in range(stSpSize)] for r in range(stSpSize)] #initialise at zero
    return Q

#Define M matrix
def init_M_matrix(nDisks): 
    stSpSize = 3**nDisks
    M = [[None for c in range(stSpSize)] for r in range(stSpSize)] #Always start with None
    return M


#Run Q learning algo with one look ahead only  - This is only used for testing purposes
def Q_learning_only_one_step_ahead(R, Q, episodes, alpha, gamma, epsilon): # epsilon not used at the moment
    Qnew = Q
    stSpSize = len(R)
    i=0
    while (i<episodes):  # need to add a matrix convergence test 
        s = random.randint(0,stSpSize-1)  # pure random state selection
        #add check for goal state - probably unnecessary since goal maps to itself
        A = [ix for ix,a  in enumerate(R[s]) if a!=None] #list of allowable actions from state s
        a = A[random.randint(0,len(A)-1)]    #choose a random action a (action is also a state in this case)
        Aa = [ix for ix,a  in enumerate(R[a]) if a!=None] #list of allowable actions from state a
        Qmax  = max([Q[a][svalid] for svalid in Aa])  # estimation policy (always greedy)
        Qnew[s][a] =Q[s][a] + alpha*(R[s][a] + gamma * Qmax - Q[s][a])
        Q = Qnew
        i +=1
    return Qnew

def Q_learning_until_goal_state(R, Q, episodes, alpha, gamma, epsilon, goal, decay, verbose=True, max_steps=0): 
    Qnew = Q
    stSpSize = len(R)
    steps = [] 
    i=0
    while i<episodes:  
        s = random.randint(0,stSpSize-1)
        j=0
        while (s !=goal):
            A = [ix for ix,a  in enumerate(R[s]) if a!=None] #list of allowable actions from state s
            rnd = random.random()
            if rnd<epsilon:  #explore - choose a random action 'a' (action is also a state in this case)
                a = A[random.randint(0,len(A)-1)]    
            else:            #exploit - choose the best action
                Qs  = [Q[s][svalid] for svalid in A]
                a = random.choice([A[ix] for ix,q in enumerate(Qs) if q == max(Qs)]) # choose randomly among the best 
            Aa = [ix for ix,a  in enumerate(R[a]) if a!=None] #list of allowable actions from state a
            Qmax  = max([Q[a][svalid] for svalid in Aa])  # estimation policy (always greedy)
            Qnew[s][a] =Q[s][a] + alpha*(R[s][a] + gamma * Qmax - Q[s][a])
            Q = Qnew 
            if decay:
                epsilon *= 0.99999 if epsilon>=0.5 else 0.9999
                #print("epsilon=", epsilon)
            s = a
            j +=1 
        i +=1
        steps.append(j)
        if verbose:
            print("\r" + "episode: " + str(i)+ "/" + str(episodes)+ " - "+str(j)
                  + "steps - Ql1norm="+str(np.linalg.norm(Q)))
    print("")
    return Qnew, steps

#Run Q learning from start state to end goal, with optional Dyna Q experience replay and optional blocking/unblocking
#This function runs until the earliest of
    #(a) maximum number of episodes
    #(b) maximum number of timesteps (the setting mostly used in our paper) unless max_steps = 0
    #(c) RMSE convergence, if early_stopping (global variable) is true

def Q_learning_until_goal_state_conv(R, Q, episodes, alpha, gamma, epsilon, goal, Qconvergence, decay, start_random=True, max_steps=0, blockage_close = [0], blockage_open = [0], verbose = False): 
    Qnew = Q
    stSpSize = len(R)

    #For blocking, define mid point [s][a] on optimal journey with n disks (used for blocking critical path):
    midpt1 = states.index(encode_state(Z,[nDisks],[n for n in range(nDisks) if n!=0],[]))
    midpt2 = states.index(encode_state(Z,[],[n for n in range(nDisks) if n!=0],[nDisks]))

    #Set-up null arrays for tracking
    steps = [] 
    steps_total=0
    trace = np.zeros(episodes) #array for RMSE trace
    trace_idx = np.zeros(episodes) #array for cumulative steps trace
    trace_R = np.zeros(episodes) #array for total reward per episode steps trace
    trace_R_bystep = np.zeros(max_steps) #array for reward per step trace

    #Reset counters
    converged = False
    i=0
    
    #Start Q learning episode from start-state until stopping criteria
    while (i<episodes and not converged and (max_steps==0 or steps_total<max_steps)):  # run until no more episodes or the Q matrix has converged
        if start_random:
            s = random.randint(0,stSpSize-1)  # purely random initial state selection
        else:
            s = 0
        j=0
        
        #Within each episode, continue Q learning until goal or stopping criteria
        while (s !=goal and (max_steps==0 or steps_total<max_steps)):
            A = [ix for ix,a  in enumerate(R[s]) if a!=None] #list of allowable actions from state s
            rnd = random.random()
            if rnd<epsilon:  #explore - choose a random action 'a' (action is also a state in this case)
                #print('random', rnd)
                a = A[random.randint(0,len(A)-1)]    
            else:            #exploit - choose the best action
                #print('greedy', rnd)
                Qs  = [Q[s][svalid] for svalid in A]
                a = random.choice([A[ix] for ix,q in enumerate(Qs) if q == max(Qs)]) # choose randomly among the best 

            #Update Model M for DynaQ learning (has no impact if n=0)
            M[s][a] = R[s][a] #store observed reward from state/action in dynaQ Model.  Note, even a 'zero' is useful here.
            for AA in range(stSpSize):
                M[s][AA] = None if R[s][AA] == None else M[s][AA] # overwrite M matrix for any past actions that are no longer valid (required for changing puzzle)

            #Standard Q learning    
            Aa = [ix for ix,a  in enumerate(R[a]) if a!=None] #list of allowable actions from state a
            Qmax  = max([Q[a][svalid] for svalid in Aa])  # estimation policy (always greedy)
            Qnew[s][a] = Q[s][a] + alpha*(R[s][a] + gamma * Qmax - Q[s][a]) #
            if steps_total < max_steps:
                trace_R_bystep[steps_total]=R[s][a] #save reward for this step, but only if capping steps
            print('Q learning action', a) if verbose else 0
            s = a # s=s'
            Q=Qnew
            steps_total = steps_total+1
            
            #implement epsilon decay
            if decay:
                epsilon *= 0.99999 if epsilon>=0.5 else 0.9999
            j +=1 
            
            #Add remove blockage at mid-point of optimum path
            if steps_total in blockage_close:
               print('blocking at ', steps_total) if verbose else 0
               R[midpt1][midpt2] = None    
            if steps_total in blockage_open:
               print('unblocking at ', steps_total) if verbose else 0
               R[midpt1][midpt2] = 0
        
            #Experience replay for Dyna-Q learning (unless n=0 i.e. normal Q learning)
            for k in range(dynaQn):
                #randomly select historic move
                historic_move = False
                rancount = 0
                while historic_move == False and rancount < 1000:
                    s_dyna = random.randint(0,stSpSize-1)
                    a_dyna = random.randint(0,stSpSize-1)
                    if M[s_dyna][a_dyna] !=None:
                        historic_move = True
                        print('found historic move after ', rancount+1, ' attempts at state/action', s_dyna, a_dyna) if verbose else 0
                    rancount = rancount + 1
                
                if historic_move == True:
                    #update Q based on experience replay
                    Aa_dyna = [ix for ix,a_dyna  in enumerate(M[a_dyna]) if a_dyna!=None] #list of allowable *known* actions from state a_dyna
                    if Aa_dyna != []: 
                        print('updating Q from experience replay.  State/action = ', s_dyna, a_dyna) if verbose else 0
                        Qmax_dyna  = max([Q[a_dyna][svalid] for svalid in Aa_dyna])  # estimation policy (always greedy)
                        Qnew[s_dyna][a_dyna] = Q[s_dyna][a_dyna] + alpha*(M[s_dyna][a_dyna] + gamma * Qmax_dyna - Q[s_dyna][a_dyna]) #update Q based on past experience

        #At end of each episode, record progress
        trace_R[i]=100*np.power(gamma,j) #total reward always 100, but adjust for gamma ^ n-steps (j)
        Qarray=np.array(Q)
        Qrmse_array = rmse(Qarray,Qconvergence) #root mean square difference between Q and convergence target
        Qrmse = np.mean(Qrmse_array)
        
        #RMSE convergence test, if early_stopping global variable is true
        if Qrmse<0.001:
            if early_stopping:
                if converged == False:
                    print('     converged, stopping Q learning') if verbose else 0
                converged = True 
            else:
                print('     converged but stopping turned off')  if verbose else 0     

        trace[i]=Qrmse
        trace_idx[i]=(j+trace_idx[i-1]) if i>0 else j
        i +=1
        steps.append(j)

    #At end of learning, record progress
    print('     Total steps taken', steps_total)
    trace_idx[trace_idx==0] = max(trace_idx)
    
    return Qnew, steps, trace, trace_idx, trace_R, trace_R_bystep


#starts agent in state s0 and outputs list of states until sGoal
def run_agent(s0, sGoal, Q, maxiter):  
    #define midpoints and check if blocked
    midpt1 = states.index(encode_state(Z,[nDisks],[n for n in range(nDisks) if n!=0],[]))
    midpt2 = states.index(encode_state(Z,[],[n for n in range(nDisks) if n!=0],[nDisks]))
    blocked = True if R[midpt1][midpt2]==None else False
    
    i=0
    s = s0
    state_list = [s]
    while (i<maxiter) and (s != sGoal):
        #print(Q[s])
        s_new = Q[s].index(max(Q[s]))
        if blocked and s == midpt1 and s_new == midpt2: #if blocked and choosing disallowed move,choose next best
            Q_without_disallowed = Q[s][:s_new]+Q[s][s_new+1:]
            s_new = Q_without_disallowed.index(max(Q_without_disallowed))
        s = s_new
        state_list.append(s)
        i +=1        
    return state_list, i




def create_paramater_mesh(alpha_from, alpha_to, alpha_n, gamma_from, gamma_to, gamma_n, epsilon_from, epsilon_to, epsilon_n):   
#    alpha_n = n if alpha_to>alpha_from else 1
#    gamma_n = n if gamma_to>gamma_from else 1
#    epsilon_n = n if epsilon_to>epsilon_from else 1
    alpha_ = np.linspace(alpha_from, alpha_to, alpha_n)
    gamma_ = np.linspace(gamma_from, gamma_to, gamma_n)
    epsilon_ = np.linspace(epsilon_from, epsilon_to, epsilon_n)   
    alpha, gamma, epsilon = np.meshgrid(alpha_, gamma_, epsilon_)    
    alpha = alpha.flatten().tolist()
    gamma = gamma.flatten().tolist()
    epsilon = epsilon.flatten().tolist()    
    return alpha, gamma, epsilon
    

#Plot graph of episodes to convergence
def convergence_graph(alpha_graph, epsilon_graph, convergence_graph, gamma_title, ceiling, amin, amax, emin, emax, title): 
    #interpolate data
    x = np.array(alpha_graph)
    y = np.array(epsilon_graph)
    z = np.array(convergence_graph)
    idx = np.all([x>=amin,x<=amax,y>=emin,y<=emax], axis=0)
    x, y, z = x[idx], y[idx], z[idx]
    z[np.isnan(z)]=ceiling #anything not converged, set to max
    z[z>ceiling]=ceiling #cap anything else at the ceiling
    xi = np.linspace(amin, amax, 100)
    yi = np.linspace(emin, emax, 100)
    zi = plt.mlab.griddata(x, y, z, xi, yi, interp='linear')
    #2D contour plot (with interpolation) - works with gaps in data
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel('Alpha')
    if decay:
        ax.set_ylabel('Epsilon')
    else:
        ax.set_ylabel('Epsilon (no decay)')
    ax.set_title(title + '\n gamma =' + str("{:.0%}".format(gamma_title)))
    plt.contourf(xi, yi, zi, 10, cmap=cm.coolwarm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('episodes')
    plt.tight_layout()
    plt.show()
    #3D trisurface + 2D contour plot (with interpolation) - works with gaps in data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, edgecolors='k', antialiased=True, alpha=0.3, cmap=cm.coolwarm)
    ax.view_init(40, 30) #degrees down, degrees left 
    cset = ax.contourf(xi, yi, zi, 10, zdir='z', offset=0, cmap=cm.coolwarm)
    ax.set_xlabel('Alpha')
    if decay:
        ax.set_ylabel('Epsilon')
    else:
        ax.set_ylabel('Epsilon (no decay)')
    ax.set_title(title + '\n gamma =' + str("{:.0%}".format(gamma_title)))
    ax.set_xlim(amin, amax)
    ax.set_ylim(emin, emax)
    ax.set_zlim(0, ceiling)
    cbar = fig.colorbar(cset)
    cbar.ax.set_ylabel('episodes')
    plt.tight_layout()
    plt.show()
    
    return 


#Plot graphs of reward / rate of reward by timestep
def step_progression_graph (R_bystep, xlimit, alpha, gamma, epsilon, dynaQ_list=[], dynaQlegend=False):

    if dynaQlearning:
        from cycler import cycler
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'k', 'gray']) +
                               cycler('linestyle', ['-', '--', '-.'])))
    else:
        from cycler import cycler
        plt.rc('axes', prop_cycle=(cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']) +
                               cycler('linestyle', ['-', '-', '-', '-'])))

    R_bystep_cumulative=np.cumsum(R_bystep, axis=0)
    
    fig = plt.figure()

    if dynaQlegend:
        legend_text=['a:'+str(round(a,2))+ ', e:'+str(round(e,2))+ ', n:'+str(round(dynaQn,2)) for a, e, dynaQn in zip(alpha,epsilon, dynaQ_list)]
    else:
        legend_text=['a:'+str(round(a,2))+ ', e:'+str(round(e,2)) for a, e in zip(alpha,epsilon)]

    plt.plot(R_bystep_cumulative) 
    plt.xlim(0,xlimit)
    ylimit = np.ceil(np.max(R_bystep_cumulative[:xlimit,:]/1000))*1000
    if ylimit<=1000:
        ylimit = np.ceil(np.max(R_bystep_cumulative[:xlimit,:])*100)/100
    plt.ylim(0,ylimit)
    plt.title('Reward v time steps')
    #plt.legend(legend_text)
    ax = fig.gca()
    ax.set_ylabel('Cumulative reward')
    ax.set_xlabel('Time Steps')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(legend_text, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    #Calculate moving average reward
    window=500
    R_bystep_movingaverage = np.zeros([len(R_bystep[:,1]), len(alpha)])
    for i in range(len(alpha)):
        R_bystep_movingaverage[:,i] = np.convolve(R_bystep[:,i], np.ones((window,))/window, mode='same')
        R_bystep_movingaverage[:int(window/2),i] = None #moving average meaningless for first 250 values
        R_bystep_movingaverage[-int(window/2):,i] = None #moving average meaningless for first 250 values

    #Plot cumulative reward against time step progression - moving average gradient
    fig = plt.figure()
    plt.plot(R_bystep_movingaverage[:,:]) 
    plt.xlim(0,xlimit)
    ylimit = np.ceil(np.max(R_bystep_movingaverage[int(window/2):len(R_bystep[:,1])-int(window/2),:]))
    if ylimit<=1:
        ylimit = np.ceil(np.max(R_bystep_movingaverage[int(window/2):len(R_bystep[:,1])-int(window/2),:])*10)/10
    plt.ylim(0,ylimit)
    plt.title('Rate of reward v time steps')
    #plt.legend(legend_text)
    ax = fig.gca()
    ax.set_ylabel('Moving average reward per time step')
    ax.set_xlabel('Time Steps')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(legend_text, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    return

#Optional functions for storing and retrieving data
def manage_data(overwrite_history, save_history, load_history, add_to_history):    
    # This function saves data in a second set of arrays or saves to / loads from file / joins two sets of data
    if overwrite_history:
        alpha_cum=alpha
        gamma_cum=gamma
        epsilon_cum=epsilon
        converged_episodes_mean_cum = converged_episodes_mean
        converged_episodes_std_cum = converged_episodes_std       
        journey_length_cum = journey_length
        journey_length_score_cum = journey_length_score
        trace_R_bystep_perm_cum = trace_R_bystep_perm
        
    if add_to_history:
        alpha_cum = (alpha_cum + alpha) 
        gamma_cum = (gamma_cum + gamma) 
        epsilon_cum = (epsilon_cum + epsilon) 
        converged_episodes_mean_cum = np.concatenate((converged_episodes_mean_cum, converged_episodes_mean))
        converged_episodes_std_cum = np.concatenate((converged_episodes_std_cum, converged_episodes_std))       
        journey_length_cum = np.concatenate((journey_length_cum, journey_length))       
        journey_length_score_cum = np.concatenate((journey_length_score_cum, journey_length_score))       
        trace_R_bystep_perm_cum = np.concatenate((trace_R_bystep_perm_cum, trace_R_bystep_perm))       

    if save_history:
        timestamp = 'Hanoi'+str(round(time.time())%10000000)
        np.save(timestamp+'A',alpha_cum)
        np.save(timestamp+'E',epsilon_cum)
        np.save(timestamp+'Cmean',converged_episodes_mean_cum)
        np.save(timestamp+'Cstd',converged_episodes_std_cum) 
        np.save(timestamp+'Jlen',journey_length_cum) 
        np.save(timestamp+'Jscr',journey_length_score_cum) 
        np.save(timestamp+'Rtrc',trace_R_bystep_perm_cum) 

    #save current
        timestamp = 'Hanoi'+str(round(time.time())%10000000)
        np.save(timestamp+'A',alpha)
        np.save(timestamp+'E',epsilon)
        np.save(timestamp+'Cmean',converged_episodes_mean)
        np.save(timestamp+'Cstd',converged_episodes_std) 
        np.save(timestamp+'Jscr',journey_length_score) 
        np.save(timestamp+'Rtrc',trace_R_bystep_perm) 

        
    if load_history:
        timestamp = input("File timestamp:")
        alpha_cum = np.load('Hanoi'+str(timestamp)+'A'+'.npy')
        epsilon_cum = np.load('Hanoi'+str(timestamp)+'E'+'.npy')
        converged_episodes_mean_cum = np.load('Hanoi'+str(timestamp)+'Cmean'+'.npy')
        converged_episodes_std_cum = np.load('Hanoi'+str(timestamp)+'Cstd'+'.npy')    
        journey_length_cum = np.load('Hanoi'+str(timestamp)+'Jscr'+'.npy')    
        trace_R_bystep_perm_cum = np.load('Hanoi'+str(timestamp)+'Rtrc'+'.npy')    
        
    return


def display_state(nDisks, Z, states, sarray, slabel, thcol='b', thwidth=10, filename='', codeDisp=False):
    #graphical state representation - save to disk
    nbrows, nbcols = 1,3
    fig, axes = plt.subplots(nrows = nbrows, ncols = nbcols, figsize = (2*nbcols,nbrows), 
                         sharey=True, sharex=True, linewidth=1)
    for i in range(3):
        add_zeros = [0 for n in range(nDisks-len(sarray[i]))]
        pole = - thwidth * np.sort(-np.array(sarray[i]+add_zeros))
        X = np.arange(len(pole))
        axes[i].barh(X, pole, color = thcol)
        axes[i].barh(X, -pole, color = thcol)
        #axes[i].axhline(linewidth=4, color="brown")        
        axes[i].axvline(linewidth=4, color="brown")
        axes[i].set_axisbelow(True)
        axes[i].axis('off')
    if codeDisp:
        sNum = encode_state(Z, sarray[0], sarray[1], sarray[2])
        axes[0].set_title(str(states.index(sNum)), position=(0.1, 0.9), horizontalalignment='right')
        axes[1].set_title(str(sNum), position=(0.5, 0.9), horizontalalignment='center')
        axes[2].set_title(str(sarray),position=(0.9, 0.9), horizontalalignment='right')
    if filename != '':
        plt.savefig(img_path+filename+'.png', bbox_inches='tight')
    plt.show()
    return axes


def save_solution_images(nDisks, Z, states, solution):
    for i in range(len(solution)):
        display_state(nDisks, Z, states, decode_state(nDisks, Z, states[solution[i]]), nDisks, thcol='b', thwidth=10, 
                  filename='sol' + str(nDisks) + 'Dsk-' + str(i), codeDisp=True)

def save_all_states_images(nDisks, Z, states):
    for s in states:
        display_state(nDisks, Z, states, decode_state(nDisks, Z, s), 3, thcol='b', thwidth=10, 
                  filename=str(nDisks) + 'DisksStates-' + str(s), codeDisp=True)
        
 
def display_graphviz(nDisks, Z, states, img=""): ######## Display a undirected graph using graphviz
    import graphviz
    #from graphviz import Digraph
    #import pygraphviz
    G = graphviz.Digraph('G', filename='THGraph', engine='neato',format='png')#,imagepath=imgpath)
    G.attr(size='20,10', sep='3', mode='KK')#,splines='ortho')
    for s in states:
        slabel = str(decode_state(nDisks, Z, s))
        if img=="":
            G.node(str(s),slabel,shape="box", nodesep='10.0')
        else:
            G.node(str(s), image=img_path+img+str(s)+".png", label="", width='1.0', height='0.3', fixedsize='true',
                   shape="box",imagescale='both', margin='0.0')
        sidxlist = next_allowed_states(nDisks, Z, states, states.index(s))
        for sidx in sidxlist:
            sidxlabel = str(decode_state(nDisks, Z, states[sidx]))
            if states[sidx]>s:
                if img=="":
                   G.node(str(states[sidx]),sidxlabel, shape="box", nodesep='10.0') 
                else:
                    G.node(str(states[sidx]),image=img_path+img+str(states[sidx])+".png", label="",  
                           width='1.0', height='0.3', fixedsize='true',
                           shape="box",imagescale='both', margin='0.0')
                    #G.edge(str(s), str(states[sidx]),  constraint='false',arrowhead='none')
                    G.edge(str(states[sidx]),str(s),constraint='false', arrowhead='none')
            #if  not G.has_edge(slabel, sidxlabel):
               #G.edge(slabel, sidxlabel)
            #print("from: ",slabel, " to: ",sidxlabel)
    return G


def display_graphnx(nDisks, Z, states): ######## Display a undirected graph using networkx
    import networkx as nx
    G = nx.DiGraph()
    G.clear()
    for s in states:
        slabel = str(decode_state(nDisks, Z, s))
        G.add_node(slabel)
        sidxlist = next_allowed_states(nDisks, Z, states, states.index(s))
        for sidx in sidxlist:
            sidxlabel = str(decode_state(nDisks, Z, states[sidx]))
            G.add_node(sidxlabel)
            if  not G.has_edge(slabel, sidxlabel):
                G.add_edge(slabel, sidxlabel)
            #print("from: ",slabel, " to: ",sidxlabel)
    return G

def length_to_solve(start0, runs):
    #calculate average length to solve puzzle over n ('runs') iterations
    stSpSize=len(R)
    counter=0
    for i in range(runs):
        if start0:
            s0 = 0
        else:
            s0 = random.randint(0,stSpSize-1)  # purely random initial state selection 
    
        solution, iters = run_agent(s0, 3**nDisks-1, Q, nDisks*20) #assume if it can't be solved in nDisks*10, it can't be solved
        counter = counter + iters
    return (counter/runs) if solution[-1]==3**nDisks-1 else -999



############# SET Q-LEARNING VARIABLES
#Associate disk i to Z[i-1] ((i-1)th prime)
Z = prime_list(nDisks)  

#Define state list: each state is a unique integer number equal to: 
#Product(over all i, Z(i)^p(i))  with p(i)in {1,2,3} is the pole number where disk i is present
states = [1]
for i in range(nDisks):
    states = [x*y for x in states for y in [Z[i], Z[i]**2, Z[i]**3]]
states.sort()

#Start Q learning from random state each time (otherwise, random start)
start_ran = False

#Initialise R and Q matrices
R = set_R_matrix(nDisks, Z, states)
Q = init_Q_matrix(nDisks)
M = init_M_matrix(nDisks)

#Default parameters
alpha = 0.8
gamma = 0.8
epsilon = 0.9
episodes = 100
decay = False#epsilon decay

############################ SINGLE Q-LEARNING RUN
print('Single Q Learning run............')
    
#Learn Q matrix    
#Q = Q_learning_only_one_step_ahead(R, Q, 10**nDisks/2, alpha, gamma, epsilon)
#Note: no dyna Q 
Q, steps= Q_learning_until_goal_state(R, Q, episodes, alpha, gamma, epsilon, 3**nDisks-1, decay)

#Solve from s0 to sGoal
s0 = 0
solution, iters = run_agent(s0, 3**nDisks-1, Q, 100)

print("solution from state #"+str(s0)+" to state #"+str(3**nDisks-1))

plt.close('all')
for i in range(len(solution)):
    try:
        display_state(nDisks, Z, states, decode_state(nDisks, Z, states[solution[i]]), nDisks, thcol='b', thwidth=10,
              filename='sol' + str(nDisks) + 'Dsk-' + str(i), codeDisp=True)
    except FileNotFoundError:
        print(solution[i],decode_state(nDisks, Z, states[solution[i]]))
print('solution in ', iters, ' steps')


#Average journey length to solve
start0=True; runs = 1 #when testing greedy journey length, specify where to start and how many times to check (runs only relevant when stochastic)
solution_length = length_to_solve(start0,runs)
if solution_length>0:
    print('     Journey length :', solution_length)
else:
    print('     Not solved')

#Count episodes to Q convergence to judge how long to run optimization for
Qconvergence = Q
Qconvergence_array=np.array(Qconvergence)
R = set_R_matrix(nDisks, Z, states)
Q = init_Q_matrix(nDisks)
M = init_M_matrix(nDisks)

Q, steps, trace, trace_idx, trace_R, trace_R_bystep = Q_learning_until_goal_state_conv(R, Q, episodes, alpha, gamma, epsilon, 3**nDisks-1, Qconvergence, decay, start_ran, verbose=False)
converged_episodes = np.argmax(trace<0.001)

############################Q LEARNING FOR MULTIPLE PARAMETERS
print()
print('Q Learning for multiple parameters............')

##Define parameter grid for search

dynaQlearning = True #VARIED
# Note: alpha = learning rate, gamma = discount rate, epsilon = epsilon-greed

#If dynaQ - create 3 scenarios with the same alpha, epsilon, gamma but varying 'n' experience replay
if dynaQlearning:
    dynaQ_params = [0,1,3] #max length 3
    alpha, gamma, epsilon = create_paramater_mesh(0.75, 0.75, 1, 0.9, 0.9, 1, 0.75, 0.75, 1) #alpha_from, alpha_to, gamma_from, gamma_to, epsilon_from, epsilon_to, n

#If normal Q learning - create 4 scenarios with the vayring alpha, epsilon but the same gamma and 'n=0' experience replay
else:
    dynaQ_params = [0] #max length 3
    dimension = 2; alpha, gamma, epsilon = create_paramater_mesh(0.25, 0.75, dimension, 0.9, 0.9, 1, 0.25, 0.75, dimension) #alpha_from, alpha_to, gamma_from, gamma_to, epsilon_from, epsilon_to, n

#If dynaQ, iterate through alpha/gamma/epsilon params and repeat for each 'n' for dyna-Q learning
dynaQ_list = []
for n in dynaQ_params:
    dynaQ_list = dynaQ_list + [n] * len(alpha)
alpha = alpha * len(dynaQ_params)
gamma = gamma * len(dynaQ_params)
epsilon = epsilon * len(dynaQ_params)

param_array = np.array([alpha, gamma, epsilon, dynaQ_list]) #for easier copy paste to excel 

## Set 'global' parameters for all runs
episodes = 10000 #max number of episodes before stopping
iterations = 5 #test on 1, train on 20: average each measurement over more than one run
count_steps = True #Count steps for results rather than episodes
max_steps = 50000#Limit on number of steps per run (0 to ignore and just cap on episodes)
journey_optimum = 2**nDisks-1 #For reference, this is optimum solution length 

#Set grid blockage open/close times in steps
blockage_close = [] #no blockage 
blockage_open = [] #no blockage
#blockage_close = [20002] #option 1 #VARIED
#blockage_open = [1,40002] #option 1 #VARIED

#Find converged state (Q matrix varies depending on gamma)
R = set_R_matrix(nDisks, Z, states)
Q = init_Q_matrix(nDisks)
M = init_M_matrix(nDisks)
print()
print('First, finding converged Q matrix to test convergence against.......')
Qconvergence, steps, trace, trace_idx, trace_R, trace_R_bystep = Q_learning_until_goal_state_conv(R, Q, episodes, 0.9, gamma[0], 0.9, 3**nDisks-1, Qconvergence, decay, start_ran, max_steps)
print()

#Setup arrays to store results of all runs
converged_episodes_mean = np.zeros(len(alpha)) #empty array for n episodes per run mean
converged_episodes_std = np.zeros(len(alpha)) #empty array for n episodes per run standard deviation
bestQrmse = np.zeros(len(alpha)) #empty array for final rmse on each run
journey_length = np.zeros((iterations,len(alpha))) #empty array to store average journey length to goal state
trace_idx_perm = np.zeros((episodes,len(alpha))) #temporary array for trace steps
trace_R_perm = np.zeros((episodes,len(alpha))) #temporary array for episode reward
trace_R_bystep_perm = np.zeros((max_steps,len(alpha))) #temporary array for episode reward

#Start timer and plot for looping through all parameters
start_time = time.time()
plt.figure
#Repeat over each set of parameters
for i in range(len(alpha)):
    print()
    dynaQn = dynaQ_list[i]
    print('Q learning scenario ',i, ': alpha:', alpha[i],' gamma:',gamma[i], ' epsilon:', epsilon[i], ' dyna-Q with n=', dynaQn)
    #create empty arrays for storing results
    trace_temp = np.zeros((min(1000000,episodes*100),iterations)) #temporary array for trace iterations through 5 iterations
    trace_idx_temp = np.zeros((episodes,iterations)) #temporary array for trace step index by episode
    trace_R_temp = np.zeros((episodes,iterations)) #temporary array for trace of reward by episode
    trace_R_bystep_temp = np.zeros((max_steps,iterations)) #temporary array for trace of reward by step
    iteration_array = np.zeros(iterations) #empty array to store n episodes to conversion 

    #Repeat over n 'iterations' for averaging
    for j in range(iterations):

        #Reset R&Q matrices 
        R = set_R_matrix(nDisks, Z, states) #reset R&Q
        Q = init_Q_matrix(nDisks)
        M = init_M_matrix(nDisks)
        #Run Q learning algorithm
        Q, steps, trace, trace_idx, trace_R, trace_R_bystep = Q_learning_until_goal_state_conv(R, Q, episodes, alpha[i], gamma[i], epsilon[i], 3**nDisks-1, Qconvergence, decay, start_ran, max_steps, blockage_close, blockage_open)
        
        #Store results matrix
        trace_new = trace
        if count_steps:
            #interpolate steps array to replace episodes with steps
            xaxis=range(0, int(episodes*250))
            trace_new = np.interp(xaxis, trace_idx, trace)
        iteration_array[j]=np.argmax(trace_new<0.01) #find index of first time RMSE below threshold, ie episodes/steps to converge
        if iteration_array[j]==0:
            iteration_array[j]=np.nan            
        trace_temp[:min(1000000, episodes*100,len(trace_new)),j]=trace_new[:min(1000000,episodes*100,len(trace_new))] #RMSE over time
        trace_idx_temp[:min(episodes,len(trace_idx)),j]=trace_idx
        trace_R_temp[:min(episodes,len(trace_idx)),j]=trace_R
        trace_R_bystep_temp[:len(trace_R_bystep),j]=trace_R_bystep
        journey_length[j][i]=length_to_solve(start0,runs)
        print('     Iteration ',j,': length to solve = ',journey_length[j][i])

    #One line on plot for each set of parameters (averaged over n iterations) 
    plt.plot(np.mean(trace_temp,axis=1)) #plot average convergence line for these parameters

    #Diagnostics
    print()
    print('     Scenario ',i,': summary for all iterations... ')
    print('     ', iteration_array, 'episodes to convergence')
    print('     ', trace_temp[-1,:], 'final RMSE')    
    print('     ', journey_length[:,i], 'journey lengths')

    #Save results
    bestQrmse[i]=round(trace[-1],2)
    converged_episodes_mean[i] = np.mean(iteration_array) 
    converged_episodes_std[i] = np.std(iteration_array) 
    trace_idx_perm[:,i]=np.mean(trace_idx_temp,axis=1)
    trace_R_perm[:,i]=np.mean(trace_R_temp, axis=1)
    trace_R_bystep_perm[:,i]=np.mean(trace_R_bystep_temp, axis=1)
    journey_length_score = np.sum(journey_length>0, axis=0)*0.5 + np.sum(journey_length==journey_optimum, axis=0)*0.5
    
elapsed_time = time.time() - start_time

#Now show graph
print()
print('RMSE between Q and Q converged over timesteps')
xlim = max_steps
plt.xlim(xmin=0, xmax=xlim)
plt.show()

#Summary results
print()
print('FINAL SUMMARY, AVERAGED OVER ',iterations,' ITERATIONS')
print(alpha, 'alpha (learning rate)')
print(gamma, 'gamma (discount rate)')
print(epsilon, 'epsilon (random/greedy)')
print(bestQrmse, 'best RMSE')
#print(converged_episodes_mean, 'episodes to convergence - mean')
#print(converged_episodes_std, 'episodes to convergence - standard deviation')
print(journey_length, 'average journey length')

##Graphs
#Plot cumulative reward and moving average rate of reward against time step progression
print()
print('STEP PROGRESSION GRAPHS')
xlimit=10000 #VARIED
step_progression_graph(trace_R_bystep_perm, xlimit, alpha, gamma, epsilon, dynaQ_list, dynaQlegend=True) #data, graph xlimit, parameters
    
#Plot graphs of cumulative results (alpha,epsilon, z, gamma, zceiling, amin, amax,emin, emax)
surface_graph=False #Note graphs assume multiple alpha/epsilon parameters run to convergence
if surface_graph: 
    ceiling = 20000 
    print()
    print('CONVERGENCE GRAPH')
    convergence_graph(alpha, epsilon, converged_episodes_mean, gamma[0], ceiling, min(alpha),max(alpha),min(epsilon),max(epsilon), 'Mean episodes to convergence')
    #convergence_graph(alpha, epsilon, journey_length_score, gamma[0], ceiling, min(alpha),max(alpha),min(epsilon),max(epsilon),'Journey Optimization Score')

print('WARNING: early stopping is on. Graphs may be affected') if early_stopping else 0