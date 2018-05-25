## Q-Learning Tower of Hanoi - Version with function approximators
# Yann Adjanor, Richard Mann, Mar 2018

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


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    filename = mypath + fig_id + "." + fig_extension
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(filename, format=fig_extension, dpi=resolution)


#Global vars
nDisks = 3
Z = [0] * (nDisks) # List of the first n prime numbers
states = [0] * (3**nDisks) # List of all possible states
alpha0=[]
epsilon0=[]
converged_episodes_mean0 = []
converged_episodes_std0 = []

#cstates = [(1,2,0),(1,0,2),(2,1,0),(2,0,1),(0,1,2),(0,2,1),(1,0,0),(0,1,0),(0,0,1)]
cstates = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0),(1,0,0),(0,1,0),(0,0,1)]
actions = [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]
W = [0.1,0.2,0.3,0.4] # function approximator weights - state = 4 params
W = [random.random(),random.random(),random.random(),random.random()] # function approximator weights
W=[0,0,0,0,0,0]
W = [0.1,0.2,0.3,0.4] # function approximator weights - state = 4 params



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

def pole_weight(Z, p1, p2, p3):    # o(n) complexity
    p = [[Z[i-1]**(np+1) for i in pn] for np, pn in enumerate((p1,p2,p3))]
    w1 = 0 if p[0]==[] else reduce(lambda x, y: x * y, p[0])
    w2 = 0 if p[1]==[] else reduce(lambda x, y: x * y, p[1])
    w3 = 0 if p[2]==[] else reduce(lambda x, y: x * y, p[2])
    return (w1,w2,w3)


#From the state integer number, get the positions of all disks
def decode_state(nDisks, Z, sNum):    #around o(nxsqrt(n)) complexity
    p = [[] for i in range(3)]
    for i in range(nDisks):
        pn = len([i for i, e in enumerate(sNum % Z[i]**p for p in [1,2,3]) if e == 0])
        p[pn-1].append(i+1)
    return p[0], p[1], p[2]


#Define macro state: a vector containing the value of the top disks in each pole
def macro_state(nDisks, Z, states, s): #returns a 3-vetor with the value of the top disks
    p1, p2, p3 = decode_state(nDisks, Z, states[s])
    #possible variation: no disk=largest possible disk
    #return (nDisks+1) if p1==[] else min(p1), (nDisks+1) if p2==[] 
    #else min(p2), (nDisks+1) if p3==[] else min(p3) 
    return [0 if p1==[] else min(p1), 0 if p2==[] else min(p2), 0 if p3==[] else min(p3)] 


#returns a 3-vector with values 0, 1 or 2 depending of the rank of the top disk on each pole
# 0 can't move (largest disk or no disk)
# 1 smallest disk
# 2 medium disk
def compact_state(nDisks, Z, states, s): 
    p1, p2, p3 = macro_state(nDisks, Z, states, s)
    idx = [i[0] for i in sorted(enumerate([p1, p2, p3]), key=lambda k: k[1])]
    [t1, t2, t3] =[idx.index(i) for i in range(3)] 
    [t1, t2, t3] =[t*p if p==0 else t for t,p in zip([t1,t2,t3],[p1,p2,p3])]  # takes care of 2 zeros case
    if sum([p1-t1, p2-t2, p3-t3])<0: # normalise to 1
        [t1, t2, t3] = [p1, p2, p3]
    if p1*p2*p3 !=0:
        [t1, t2, t3] =[0 if t==2 else t+1 for t in [t1,t2,t3]]  # takes care of no zeros case
    return (t1, t2, t3)


#returns a 3-vector with values 0, 1 or 2 depending of the rank of the top disk on each pole
# 0 no disk
# 1 smallest disk
# 2 medium disk
# 3 largest disk this time differentiate with o
############# not implemented yet
def less_compact_state(nDisks, Z, states, s): 
    p1, p2, p3 = macro_state(nDisks, Z, states, s)
    idx = [i[0] for i in sorted(enumerate([p1, p2, p3]), key=lambda k: k[1])]
    [t1, t2, t3] =[idx.index(i) for i in range(3)] 
    [t1, t2, t3] =[t*p if p==0 else t for t,p in zip([t1,t2,t3],[p1,p2,p3])]  # takes care of 2 zeros case
    if sum([p1-t1, p2-t2, p3-t3])<0: # normalise to 1
        [t1, t2, t3] = [p1, p2, p3]
    if p1*p2*p3 !=0:
        [t1, t2, t3] =[0 if t==2 else t+1 for t in [t1,t2,t3]]  # takes care of no zeros case
    return (t1, t2, t3)

# actions are represented as 2-tuples (pi,pj) = move from pole pi to pole pj with pi,pj in [1,2,3]
def next_state(nDisks, Z, states, s, a):
    t = macro_state(nDisks, Z, states, s)
    #print('disk: ',t[a[0]-1], ', Z=',Z[t[a[0]-1]], ', Exponent=',  (a[1]-a[0]))
    sprimenum = states[s]*(Z[t[a[0]-1]-1]**(a[1]-a[0]))  #must be a valid move
    return states.index(round(sprimenum))  #returns state index


#Transition function 
# actions are represented as 2-tuples (pi,pj) move from pole pi to pole pj pi,pj in [1,2,3]
def next_allowed_actions(nDisks, Z, states, s):
    cs  = compact_state(nDisks, Z, states, s)
    A=[]
    if cs == (0,0,1):   #Goal state 
        A = []        #this might be too restrictive  examine potential change
    else:
        #print(sNum)
        for i in range(3): # examine each pole in succession
            # compare top of pole i with pole i+1, if lower or empty, move there
            if (cs[i]!=0) and ((cs[i] < cs[(i+1)%3]) or cs[(i+1)%3]==0): 
                A.append((i+1,(i+1)%3+1))
            # compare top of pole i with pole i+2, if lower or empty, move there
            if (cs[i]!=0) and ((cs[i] < cs[(i+2)%3]) or cs[(i+2)%3]==0): 
                A.append((i+1,(i+2)%3+1))    
    return A


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
                A.append(states.index(round(sNum*(Z[t[i]-1]**((i+2)%3-i))))) 
    return A


def approx_state(nDisks, Z, states, s): #function converting state s into its features
    return compact_state(nDisks, Z, states, s) #initially just look at compact state approximation
    #cs = macro_state(nDisks, Z, states, s) # value of the top disks as approximation
    #cs.append(nDisks)                      #add total number of disks as additional feature
    #return cs

#function approximator: returns state approximator for state reached from state s after taking action a
#in our case this is the same as the compact state (but could be modified)
#returns a vector of features described in 
def f_approx(nDisks, Z, states, s, a): # s: state index, a: 2-tuple action 
    return f4(nDisks, Z, states, s, a, normalise=True)


def f1(nDisks, Z, states, s, a, normalise=True): # values of compact state - plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = compact_state(nDisks, Z, states, sprime)
    #f = [1 if cs==cstates[i] else 0 for i in range(len(cstates))]
    f = [1 if (cs==cstates[i] and a==actions[j]) else 0 for i in range(len(cstates)) for j in range(len(actions))]
    f.append(1) #add bias
    #print(approx_state(nDisks, Z, states, s),a,cs, f)
    return f


def f2(nDisks, Z, states, s, a, normalise=True): #values of compact state - plus pole height normalised - plus actions 
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = compact_state(nDisks, Z, states, sprime)
    cs = [cs[0],cs[1],cs[2]]
    #f = [1 if cs==cstates[i] else 0 for i in range(len(cstates))]
    p1,p2,p3 = decode_state(nDisks, Z, states[s]) # not effcient but for clarity
    cs.append(len(p1)/nDisks) #add p1 relative height
    cs.append(len(p2)/nDisks) #add p2 relative height
    cs.append(len(p3)/nDisks) #add p3 relative height
    f = [1 if (cs==cstates[i] and a==actions[j]) else 0 for i in range(len(cstates)) for j in range(len(actions))]
    f.append(1) #add bias
    #print(approx_state(nDisks, Z, states, s),a,cs, f)
    return f

def f3(nDisks, Z, states, s, a, normalise=True): # values of top disks - plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = macro_state(nDisks, Z, states, sprime)
    f = [cs[i] if (a==actions[j]) else 0 for i in range(3) for j in range(len(actions))]
    f.append(1) #add bias
    return f

    
#######works with N=3 alpha=0.05 gamma=0.8 epsilon=0.5 episodes=2000 decay=True, and 0 start always
def f4(nDisks, Z, states, s, a, normalise=True): # values of top disks and pole heights - normalised - plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = macro_state(nDisks, Z, states, sprime)
    p1,p2,p3 = decode_state(nDisks, Z, states[sprime]) # not effcient but for clarity
    #cs.append(a[0]) # add action
    #cs.append(a[1]) # add action
    #cs.append(1) # add bias
    cs.append(len(p1)) #add p1  height
    cs.append(len(p2)) #add p2  height
    cs.append(len(p3)) #add p3  height
    if normalise:
        f = [cs[i]/nDisks if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise
    else:
        f = [cs[i] if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise    
    f.append(1) #add bias
    return f

#######works with N=3 alpha=0.05 gamma=0.8 epsilon=0.4 episodes=2000 decay=True, and 0 start always
def f4_ext(nDisks, Z, states, s, a, normalise=True): # values of top disks and pole heights - normalised - plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = macro_state(nDisks, Z, states, sprime)
    p1,p2,p3 = decode_state(nDisks, Z, states[sprime]) # not effcient but for clarity
    #cs.append(a[0]) # add action
    #cs.append(a[1]) # add action
    #cs.append(1) # add bias
    cs.append(len(p1)) #add p1  height
    cs.append(len(p2)) #add p2  height
    cs.append(len(p3)) #add p3  height
    if normalise:
        f = [cs[i]/nDisks if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise
    else:
        f = [cs[i] if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise    
    for i in range(nDisks):
        f.append(1) #add N bias
    return f


def f5(nDisks, Z, states, s, a, normalise=True): # values of weights - normalised - plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = macro_state(nDisks, Z, states, sprime) #[]
    p1,p2,p3 = decode_state(nDisks, Z, states[sprime]) # not effcient but for clarity
    w1,w2,w3 = pole_weight(Z, p1, p2, p3)
    if normalise:
        norm = nDisks/states[3**nDisks-1]
    else:
        norm = 1
    cs.append(w1*norm) #add p1  weight
    cs.append(w2*norm) #add p2  height
    cs.append(w3*norm) #add p3  height
    #print(nDisks,states[nDisks], w1,w2,w3,cs)
    if normalise:
        f = [cs[i]/nDisks if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise
    else:
        f = [cs[i] if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] 
    f.append(1) #add bias
    return f


def f6(nDisks, Z, states, s, a, normalise=True): # values of top disks and pole heights and weights - normalised - plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = macro_state(nDisks, Z, states, sprime)
    p1,p2,p3 = decode_state(nDisks, Z, states[sprime]) # not effcient but for clarity
    w1,w2,w3 = pole_weight(Z, p1, p2, p3)
    if normalise:
        norm = nDisks/states[3**nDisks-1]
    else:
        norm = 1
    cs.append(len(p1)) #add p1  height
    cs.append(len(p2)) #add p2  height
    cs.append(len(p3)) #add p3  height
    norm = nDisks/states[3**nDisks-1]
    cs.append(w1*norm) #add p1  weight
    cs.append(w2*norm) #add p2  height
    cs.append(w3*norm) #add p3  height
    if normalise:
        f = [cs[i]/nDisks if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise
    else:
        f = [cs[i] if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] 
    f.append(1) #add bias
    return f

def f6_ext(nDisks, Z, states, s, a, normalise=True): # values of top disks and pole heights and weights - normalised - plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = macro_state(nDisks, Z, states, sprime)
    p1,p2,p3 = decode_state(nDisks, Z, states[sprime]) # not effcient but for clarity
    w1,w2,w3 = pole_weight(Z, p1, p2, p3)
    if normalise:
        norm = nDisks/states[3**nDisks-1]
    else:
        norm = 1
    cs.append(len(p1)) #add p1  height
    cs.append(len(p2)) #add p2  height
    cs.append(len(p3)) #add p3  height
    norm = nDisks/states[3**nDisks-1]
    cs.append(w1*norm) #add p1  weight
    cs.append(w2*norm) #add p2  height
    cs.append(w3*norm) #add p3  height
    if normalise:
        f = [cs[i]/nDisks if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise
    else:
        f = [cs[i] if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] 
    for i in range(nDisks):
        f.append(1) #add N bias
    return f


def f7(nDisks, Z, states, s, a, normalise=True): # values of top disks and pole heights and weights - normalised - plus parity of N, plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = macro_state(nDisks, Z, states, sprime)
    p1,p2,p3 = decode_state(nDisks, Z, states[sprime]) # not effcient but for clarity
    w1,w2,w3 = pole_weight(Z, p1, p2, p3)
    if normalise:
        norm = nDisks/states[3**nDisks-1]
    else:
        norm = 1
    cs.append(len(p1)) #add p1  height
    cs.append(len(p2)) #add p2  height
    cs.append(len(p3)) #add p3  height
    norm = nDisks/states[3**nDisks-1]
    cs.append(w1*norm) #add p1  weight
    cs.append(w2*norm) #add p2  height
    cs.append(w3*norm) #add p3  height
    if normalise:
        f = [cs[i]/nDisks if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise
    else:
        f = [cs[i] if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] 
    f.append(1) #- 2*(nDisks % 2))  #add bias (1 if even -1 if odd)
    return f

def describe_f8(actions):
    descr = []
    descr.append('h1')
    descr.append('h2')
    descr.append('h3')
    descr.append('g1')
    descr.append('g2')
    descr.append('g3')
    return [(j+6*i, descr[i], actions[j]) for i in range(len(descr)) for j in range(len(actions))] 

def f8(nDisks, Z, states, s, a, normalise=True): # pole heights and weights - normalised - plus parity of N, plus actions
    sprime = next_state(nDisks, Z, states, s, a) #state after taking action a
    cs = macro_state(nDisks, Z, states, sprime) # do not add actual states
    p1,p2,p3 = decode_state(nDisks, Z, states[sprime]) # not effcient but for clarity
    w1,w2,w3 = pole_weight(Z, p1, p2, p3)
    if normalise:
        norm = nDisks/states[3**nDisks-1]
    else:
        norm = 1
    cs.append(1 if len(p1)==0 else 1/len(p1)) #add p1  height
    cs.append(1 if len(p2)==0 else 1/len(p2)) #add p2  height
    cs.append(1 if len(p3)==0 else 1/len(p3)) #add p3  height
    norm = nDisks/states[3**nDisks-1]
    cs.append(1 if w1==0 else 1/w1) #add p1  weight
    cs.append(1 if w2==0 else 1/w2) #add p2  height
    cs.append(1 if w3==0 else 1/w3) #add p3  height
    if normalise:
        f = [cs[i] if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] #normalise
    else:
        f = [cs[i] if (a==actions[j]) else 0 for i in range(len(cs)) for j in range(len(actions))] 
    f.append(1)  #add bias 
    #for i in range(nDisks):
    #    f.append(1) #add N bias
    #print("s=",s," cs=",cs, "w1w2w3", w1,w2,w3)
    return f


#######works with N=4,5 alpha = 0.05 gamma = 0.8 epsilon = 0.5  episodes = 5000 decay = True maxsteps=1000
def f9(nDisks, Z, states, s, a, normalise=True): # ALL the data = equivalent to standard Q-learning
    #s= sprime = next_state(nDisks, Z, states, s, a) #state after taking action a - less
    #print(nDisks,states[nDisks], w1,w2,w3,cs)
    f = [1 if ((s==i) and (a==actions[j])) else 0 for i in range(len(states)) for j in range(len(actions))] #normalise
    f.append(1) #add bias
    return f

#R and Q are now defined as functions, not matrices
#Define R 
def R(nDisks, Z, states, s):  # reward to be in state s (irrespective of a...)
    return 100 if s == (3**nDisks-1) else 0


#Define Q as the function: Q(s,a)= sum/i W[i]*f[i]
def Q(nDisks, Z, W, states, s, a):
    return sum([w*cs for w,cs in zip(W,f_approx(nDisks, Z, states, s, a))])


def init_W(n, rnd=True):
    return [random.random() if rnd else 0 for x in range(n)]

    

def Q_learning_with_approximators(nDisks, Z, W, states, episodes, alpha, gamma, epsilon, sGoal, decay, verbose=True, maxsteps=1000): 
    steps = [] 
    Wnorm = []
    eps = []
    W_ts = []
    i=0
    while (i<episodes) :  
        s = 0 #random.randint(0,len(states)-1)  #start with random state
        #print('Initial state:', decode_state(nDisks, Z, states[s]))
        j=0
        while (s !=sGoal) and (j<=maxsteps):
            #print('current state s:', decode_state(nDisks, Z, states[s]))
            A = next_allowed_actions(nDisks, Z, states, s) #list of allowable actions from state s (represented as compact state cs)
            #print('    allowable actions from s:', A)
            rnd = random.random()
            if rnd<epsilon:  #explore - choose a random action 'a'
                a = A[random.randint(0,len(A)-1)]    
            else:            #exploit - choose the best action
                Qsa  = [Q(nDisks, Z, W, states, s, x) for x in A] #f_approx
                #print('Qsa:', Qsa)
                a = random.choice([A[ix] for ix,q in enumerate(Qsa) if q == max(Qsa)]) # choose randomly among the best 
            #print('    action chosen:',a)
            sprime = next_state(nDisks, Z, states, s, a)
            #print('    sprime:', decode_state(nDisks, Z, states[sprime]))
            Aa = next_allowed_actions(nDisks, Z, states, sprime) #list of allowable actions from state sprime
            #print('        allowable actions from sprime:', Aa)            
            Qmax  = 0 if Aa==[] else max([Q(nDisks, Z, W, states, sprime, x) for x in Aa])  # estimation policy (always greedy)
            escalar =  alpha*(R(nDisks, Z, states, sprime) + gamma * Qmax - Q(nDisks, Z, W, states, s, a))
            fvector = f_approx(nDisks, Z, states, s, a) # state function vector
            #print('        f vector', fvector)
            Werror = [escalar*x for x in fvector] #calculate error vector
            W = [sum(x) for x in zip(W, Werror)]  # add error term to each weight
            normW = np.linalg.norm(W)
            #W = [x/normW for x in W] ######## normalise weights - maybe not
            s = sprime
            j +=1 
            #print('Qmax:',Qmax,' err: ', escalar, ' new W:',["%0.2f" % i for i in W], ' Error:', ["%0.2f" % i for i in error])
            #print('    ->end of step ', j)
        i +=1
        if decay:
            epsilon *= 0.99999 if epsilon>=0.5 else 0.9999
            #epsilon *= 0.9999 if epsilon>=0.5 else 0.999
            #print("epsilon=", epsilon)
        #print('--->end of episode ',i)
        #print('Qmax:',Qmax,' err: ', escalar, ' new W:',["%0.2f" % i for i in W], ' Error:', ["%0.2f" % i for i in error])
        steps.append(j)
        Wnorm.append(normW)
        eps.append(epsilon)
        W_ts.append(W)
        if verbose:
            sys.stdout.write("\r" + "episode: " + str(i)+ "/" + str(episodes)+ " - "+str(j)+" steps- W norm=" \
                             +str(normW)+" epsilon:"+str(epsilon))
            sys.stdout.flush()
            #print("\r" + "episode: " + str(i)+ "/" + str(episodes)+ " - "+str(j)
            #      + "steps - Wl1norm="+str(np.linalg.norm(W)))
    #print("\n")
    return W, steps, Wnorm, eps, W_ts


        #print("\r" + "episode: " + str(i)+ "/" + str(episodes)+ " - "+str(j)
        #      + "steps - Ql1norm="+str(np.linalg.norm(Q)))
        #sys.stdout.write("\r" + "episode: " + str(i)+ "/" + str(episodes)+ " - "+str(j)+ "steps")
        #sys.stdout.flush()


def run_agent(nDisks, Z, W, states, s0, sGoal, maxiter):  #starts agent in state s0 and outputs list of states until sGoal
    i=0
    s = s0
    state_list = [s]
    while (i<maxiter) and (s != sGoal):
        A = next_allowed_actions(nDisks, Z, states, s)
        Qsa  = [Q(nDisks, Z, W, states, s, a) for a in A] #f_approx
        a = random.choice([A[ix] for ix,q in enumerate(Qsa) if q == max(Qsa)]) # choose randomly among the best 
        s = next_state(nDisks, Z, states, s, a)
        state_list.append(s)
        #print('Qsa:', Qsa)
        i +=1        
    return state_list, i





############# SET Q-LEARNING VARIABLES
#Associate disk i to Z[i-1] ((i-1)th prime)
Z = prime_list(nDisks)  

#Define state list: each state is a unique integer number equal to: 
#Product(over all i, Z(i)^p(i))  with p(i)in {1,2,3} is the pole number where disk i is present
states = [1]
for i in range(nDisks):
    states = [x*y for x in states for y in [Z[i], Z[i]**2, Z[i]**3]]
states.sort()

w0=f_approx(nDisks, Z, states, 0, (1,2))
#W = init_W(1+len(cstates)*len(actions))
W = init_W(len(w0),rnd=True)
print('W length:', len(W))
print('NDisks :', nDisks)
print(describe_f8(actions))

#Default parameters
alpha = 0.05 #0.05
gamma = 0.8
epsilon = 0.5 #0.5
episodes = 5000 #5000
decay = True #epsilon decay #True
############################
#works with N=4,5 alpha = 0.05 gamma = 0.8 epsilon = 0.5  episodes = 5000 decay = True maxsteps=1000
s0 = 0
sGoal = 3**nDisks-1
maxiter = 10000
#Q-Learning
W, steps, Wnorm, eps, W_ts = Q_learning_with_approximators(nDisks, Z, W, states, episodes, alpha, gamma, epsilon,
                                         sGoal, decay, verbose=True, maxsteps=1000)



#Solve from s0 to sGoal 
print("\n")
solution, iters = run_agent(nDisks, Z, W, states, s0, sGoal, maxiter)
print(' Weights:', W)
if iters == maxiter:
    print("no solution found")
    for i in range(10):
        print(decode_state(nDisks, Z, states[solution[i]]))
else:
    print("solution from state #"+str(s0)+" to state #"+str(3**nDisks-1)+ " in "+str(iters)+" steps")
    for i in range(len(solution)):
        print(decode_state(nDisks, Z, states[solution[i]]))




plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

fig , axes = plt.subplots(3,1, figsize=(40, 21))

axes[0].plot(steps,'orange')
axes[0].set_title("Number of steps per episode(RHS) and W-norm(black-LHS)", fontsize=20)

ax2 = axes[0].twinx()
ax2.plot(Wnorm,'k', alpha=0.6,linewidth=5)

for i in range(len(W)):
    axes[1].plot(range(episodes),[W_ts[x][i] for x in range(len(W_ts))]) 
    axes[1].annotate('w%s' %i, xy=((episodes-i),W_ts[len(W_ts)-1][i]), textcoords='data', fontsize=14)
axes[1].set_title("Weights Evolution(RHS) and Epsilon(red-LHS)", fontsize=20)

ax3 = axes[1].twinx()
ax3.plot(eps,'r',linewidth=5)

axes[2].bar([x for x in range(len(W))], W)
axes[2].set_xticks(np.arange(0, len(W), 100))
axes[2].set_title("Final Weights", fontsize=20)
#axes[2].set_xticks([int(x * 100) for x in range(len(W)/100)])
l = axes[2].set_xticklabels(labels=['w'+str(int(x*100)) for x in range(int(len(W)/100))], rotation = 90)

save_fig('F4-Nequal3', tight_layout=True, fig_extension="png", resolution=150)

