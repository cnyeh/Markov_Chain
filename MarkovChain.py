import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def train_Q(session):
    ses_train = session[n_t:]
    train_state, counts = np.unique(ses_train, return_counts=True)
    Q = np.zeros(m)  # initial distribution qi
    for i in range(m):
        if total_state[i] in train_state:
            Q[i] = counts[np.where(train_state == total_state[i])]
    if np.sum(Q) != 0:
        Q = Q / np.sum(Q)
    Q[Q == 0] = 1e-6
    return Q

def train_P(session):
    ses_train = session[n_t:]
    P = np.zeros((m, m))
    for k in range(ses_train.shape[0] - 1):
        # i: state index of eve_train[k] in total_state
        i = np.where(total_state == ses_train[k])[0][0]
        # j: state index of eve_train[k+1] in total_state
        j = np.where(total_state == ses_train[k + 1])[0][0]
        # P[i,j]: transition matrix from state i to state j
        P[i, j] = P[i, j] + 1
    for k in range(m):
        row_sum = np.sum(P[k, :])
        if row_sum != 0:
            # normalization
            P[k, :] = P[k, :] / row_sum
    P[P == 0] = 1e-6
    return P

def prob_stream(session):
    prob = []
    for k in range(session.shape[0] - T):
        # start: starting state (i.e. session[k])
        start = np.where(total_state == session[k])[0][0]
        #
        log_prob = np.log10(Q[start])
        for x in range(T):
            # current state
            i = np.where(total_state == session[k + x])[0][0]
            # next state
            j = np.where(total_state == session[k + x + 1])[0][0]
            # log transisiton probability
            log_prob = log_prob + np.log10(P[i, j])
        prob.append(log_prob)
    return prob

def session_average(session):
    start = np.where(total_state == session[0])[0][0]
    log_prob = np.log10(Q[start])
    for x in range(np.shape(session)[0]-1):
        i = np.where(total_state == session[x])[0][0]
        j = np.where(total_state == session[x+1])[0][0]
        log_prob = log_prob + np.log10(P[i,j])
    return 10**(log_prob)


def check_miss_state(session):
    possible_state = np.unique(session)
    s = 0
    for i in range(np.shape(possible_state)[0]):
        x = np.where(train_state == session[i])
        if np.shape(x[0])[0] == 0:
            s = 1
            break
    return s





path = '/Users/CanonYeh/Dropbox/audit_bsm_mill/'
#sid = [0, 2025, 2363, 2379, 2565, 2582, 2590, 2755, 2760, 2765, 2777, 2790, 2802, 2806, 2839, 2914, 2944, 2069, 3078, 3108, 3151, 3312]
attack_id = np.array([8,9,10,12,13,14,16])
attack_sess = np.array([2755, 2760, 2765, 2790, 2802, 2806, 2914]);

#read in all session ids
sid = []
file = open(path + 'sessions_and_pids.text')
f = file.readline()
while f:
    f = f.split()
    s = int(f[1])
    sid.append(s)
    f = file.readline()
sid = np.array(sid)

eve = [] #events in whole data
t = [] #time in whole data
a_type = [] #event types in whole data
a_stamp = [] #time stamps in whole data
event_ = [] #event in whole data for different sessions
time_ = [] #time in whole data for different sessions

for i in range(0, sid.size):
    file = open(path + str(sid[i]) + '.log')
    f = file.readline()
    #print f
    # event of each session
    eve_ = []
    # time of each session
    t_ = []
    while f:
        f = f.split(',')
        for j in range(0, len(f)-1):
            # event type is right before the space
            # time is right after the space
            if f[j] == '':
                t.append(pd.to_datetime(f[j+1]))
                t_.append(pd.to_datetime(f[j+1]))
                break
        ev = f[3]
        eve.append(ev)
        eve_.append(ev)
        #eve.append(ev.rstrip(',')) #if don't want the ',' at the end of event type, use this one
        #eve_.append(ev.rstrip(','))

        #update event type and time stamp
        if (eve[len(eve) - 1] in a_type) == False:
            a_type.append(eve[len(eve) - 1])
        if (t[len(t) - 1] in a_stamp) == False:
            a_stamp.append(t[len(t) - 1])
        f = file.readline()
    event_.append(eve_)
    time_.append(t_)

eve = np.array(eve) #event series of whole data
t = np.array(t) #time series of whole data
index = np.argsort(t)
t = t[index]   # time series of whole data in order
eve = eve[index]  # event series of whole data in order
a_type = np.array(a_type) #event type, not sorted
a_stamp = np.array(a_stamp) #maybe not needed

s_time = t[0] #start time
e_time = t[t.size-1] #end time
# training start time: an hour before end time
t_s_time = e_time - datetime.timedelta(0, 0, 0, 0, 0, 1, 0) #use the last hour as training data
n_t = np.where(t >= t_s_time)
n_t = n_t[0][0]; #corresponding n of the start time of training data

eve_train = eve[n_t:] # extract training data
total_state = np.unique(eve) # total possible states in a sorted order
m = np.unique(eve).shape[0] # dimension of the transition matrix
Q = np.zeros(m)
P = np.zeros((m,m))

#%% Train Q and P for each session individually and take average
count = 0.0
for i in range(np.shape(sid)[0]):
    session = event_[i]
    session = np.array(session)
    time = time_[i]
    time = np.array(time)
    n_t = np.where(time >= t_s_time)
    if np.shape(n_t[0])[0] != 0:
        n_t = n_t[0][0]
        count+=1
        Q = Q + train_Q(session)
        P = P + train_P(session)
Q = Q/count
P = P/count

## Train Q and P for all session at the same time
#Q = train_Q(eve)
#P = train_P(eve)

### Prediction for each session
T = 50 # time window to consider
plt.figure(1)
fig_count = 0
ses_prob_list = np.zeros(np.shape(sid)[0])
for i in range(np.shape(sid)[0]):
    session = event_[i]
    session = np.array(session)
    if np.shape(session)[0] < 2*T:
        ses_prob_list[i] = (session_average(session))**(1.0/np.shape(session)[0])
    else:
        fig_count+=1
        prob = prob_stream(session)
        prob = np.array(prob)
        ax = plt.subplot(5,5,fig_count)
        plt.plot(np.arange(prob.shape[0]), prob)
        ax.set_title(str(sid[i]))
        prob = 10**(prob)
        prob = prob**(1.0/T)
        ses_prob_list[i] = np.average(prob)

#%% Calculate the probability of stream of event in the whole data set
prob = prob_stream(eve)


# Prediction for whole data
plt.figure(2)
prob = np.array(prob)
plt.plot(np.arange(prob.shape[0]), prob)


# Prediction for training data
prob_train = prob_stream(eve_train)
plt.figure(3)
prob_train = np.array(prob_train)
plt.plot(np.arange(prob_train.shape[0]), prob_train)

## Session and training data averages
prob_train_real = 10**prob_train
prob_train_real = prob_train_real**(1.0/T)
print("Average probability of training data is", np.average(prob_train_real))
print("Average probability of each session is", ses_prob_list)

x = np.arange(22)
xticks = ['0', '2025', '2363', '2379', '2565', '2582', '2590', '2755', '2760', '2765', '2777', '2790', '2802', '2806', '2839', '2914', '2944', '2069', '3078', '3108', '3151', '3312']
plt.figure(4)
plt.xticks(x, xticks, rotation=45)
plt.plot(x,ses_prob_list)
plt.hlines(np.average(prob_train_real),0,21,color='r')
plt.xlim(0,21)
plt.xlabel("session ID")
plt.ylabel("Average Prob")
plt.show()
