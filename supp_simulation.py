import nest
import numpy as np

# Neuron
delay = 1.5           # Synaptic delay [ms]
C_m = 60.0      
tau_m = 60.0          # Membrane time constant [mV]
tau_psc = 1.5
V_th = 20.0          # Spike threshold [mV]
t_ref = 2.0           # Refractory period [ms]
E_L = 0.0           # Resting potential [mV]
V_reset = 10.0           # Reset potential [mV]

# Noise input
mean_noise = 12.       # Mean noisy input to all neurons [pA]
std_noise = 20.       # Std noisy input to all neurons[pA]

p_rate = 10.

# Network connectivity
N_pre_network = 100    # Number of neurons in pre synaptic network
N_post_network = 100    # Number of neurons in post synaptic network

# Histogram of number of synapses formed by bouton (MSB) from CA3 to SO in CA1 (from paper)
so = np.array([924+467,430+211,70+113,28+19,12+9])
# Synapse distribution from CA3 to SR in CA1
percentage_msb = 0.25
sr = np.zeros(len(so))
sr[0] = int((1-percentage_msb)*np.sum(so))
so_mult = so[1:]/np.sum(so[1:])
sr[1:] = (percentage_msb*np.sum(so)*so_mult).astype(int)
synapse_distribution = {'so': so, 'sr': sr}


J_E = 5.* C_m / tau_psc  # Excitatory synaptic strength [pA] (PSC max amplitude)

# Simulation parameters
simtime = 51000.        # Simulation time
rectime = 50000.        # Time considered for spiking statistics

n_simulations = 100     # Number of simulation repetitions

binsize = 20.         # Binsize for spike trains when calculating correlation coefficient [ms]
time_bins = np.arange(simtime-rectime,simtime+binsize,binsize)  # Time bins of spike trains for calculating correlation coefficient

# STF
tau_rec_STF = 130.      # Recovery time [ms]
tau_fac_STF = 530.      # Facilitation time [ms]
U_STF = 0.03      # Facilitation parameter U
A_STF = 1540.     # PSC weight [pA]

# STD
tau_rec_STD = 800.      # Recovery time [ms]
tau_fac_STD = 0.        # Facilitation time [ms]
U_STD = 0.5       # Facilitation parameter U
A_STD = 250.      # PSC weight [pA]

boutons_dict = {'so': 5,
                'sr': 7}

def n_synapses_mult(srso):
    # Selects random number of synapses formed per bouton based on experimental distribution
    syn_distribution = synapse_distribution[srso]
    syn_density = syn_distribution/np.sum(syn_distribution)
    return np.random.choice(np.arange(len(syn_distribution))+1,p=syn_density,replace=True)

def n_synapses_msb(srso):
    return boutons_dict[srso]
   
def n_synapses_ssb(srso):
    return 1
 
def select_synapse_type(stp):
    if stp==True:
        return select_STP_type()
    else:
        return 'excitatory'
        
def select_n_boutons(mult,sb_type,srso):
    if (sb_type=='MSB') & (mult==False):
        return 1
    else:
        return boutons_dict[srso]
    
def pr_constant():
    return 1
    
def pr_gamma():
    return min(np.random.gamma(shape=2,scale=0.15),1)

def select_STP_type():
    random = np.random.uniform(low=0,high=1)
    if random<0.5:
        syn_spec = 'STF'
    else:
        syn_spec = 'STD'
    return syn_spec
    
def select_n_synapses(mult,sb_type,srso):
    if mult==True:
        if sb_type=='SSB':
            n_synapses = n_synapses_ssb
        elif sb_type=='MSB':
            n_synapses = n_synapses_mult
    else:
        if sb_type=='SSB':
            n_synapses = n_synapses_ssb
        elif sb_type=='MSB':
            n_synapses = n_synapses_msb
    return n_synapses

def simulate_network_pre(master_seed):
    nest.ResetKernel()
    
    # Set parameters of the NEST simulation kernel ----------------
    nest.SetKernelStatus({'print_time' : True,
                          'local_num_threads' : 1,
                          'grng_seed' : master_seed,
                          'rng_seeds' : (master_seed+1,)})

    nest.SetDefaults('iaf_psc_exp', 
                    {'tau_m' : tau_m,
                     't_ref' : t_ref,
                     'tau_syn_ex' : tau_psc,
                     'tau_syn_in' : tau_psc,
                     'C_m' : C_m,
                     'V_reset' : V_reset,
                     'E_L' : E_L,
                     'V_m' : E_L,
                     'V_th' : V_th})

    # Create nodes -------------------------------------------------
    poisson_generator = nest.Create('poisson_generator',params={'rate':p_rate})
    neurons = nest.Create('parrot_neuron',N_pre_network)
    spike_detector = nest.Create('spike_detector')

    # Connect nodes ------------------------------------------------
    nest.Connect(poisson_generator,neurons)
    nest.Connect(neurons,spike_detector)

    # Simulate -----------------------------------------------------
    nest.Simulate(simtime)
    
    # Read data ----------------------------------------------------
    events = nest.GetStatus(spike_detector,'events')[0]
    
    return events

def simulate_network_post(master_seed,pre_events,mult,stp,pr,sb_type_so,sb_type_sr):
            
    n_boutons_so = select_n_boutons(mult,sb_type_so,'so')
    n_synapses_so = select_n_synapses(mult,sb_type_so,'so')
    
    n_boutons_sr = select_n_boutons(mult,sb_type_sr,'sr')
    n_synapses_sr = select_n_synapses(mult,sb_type_sr,'sr')
            
    if pr==True:
        pr_function = pr_gamma
    else:
        pr_function = pr_constant

    times_pre = pre_events['times']
    senders_pre = pre_events['senders']
    
    nest.ResetKernel()
    # Set parameters of the NEST simulation kernel ------------------
    nest.SetKernelStatus({'print_time' : True,
                          'local_num_threads' : 1,
                          'grng_seed' : master_seed,
                          'rng_seeds' : (master_seed+1,)})

    nest.SetDefaults('iaf_psc_exp', 
                    {'tau_m' : tau_m,
                     't_ref' : t_ref,
                     'tau_syn_ex' : tau_psc,
                     'tau_syn_in' : tau_psc,
                     'C_m' : C_m,
                     'V_reset' : V_reset,
                     'E_L' : E_L,
                     'V_m' : E_L,
                     'V_th' : V_th})

    # Create nodes -------------------------------------------------
    neurons = nest.Create('iaf_psc_exp', N_post_network)
    noise = nest.Create('noise_generator', params={'mean': mean_noise, 'std': std_noise})
    spike_detector = nest.Create('spike_detector')
    
    # Connect nodes ------------------------------------------------
    nest.CopyModel('tsodyks_synapse', 'STF',
                  {'tau_psc' : tau_psc,
                   'tau_rec' : tau_rec_STF,
                   'tau_fac' : tau_fac_STF,
                   'U' : U_STF,
                   'delay' : delay,
                   'weight' : A_STF,
                   'u' : 0.0,
                   'x' : 1.0})
    
    nest.CopyModel('tsodyks_synapse', 'STD',
                  {'tau_psc' : tau_psc,
                   'tau_rec' : tau_rec_STD,
                   'tau_fac' : tau_fac_STD,
                   'U' : U_STD,
                   'delay' : delay,
                   'weight' : A_STD,
                   'u' : 0.0,
                   'x' : 1.0})   
                   
    nest.CopyModel('static_synapse',
               'excitatory',
               {'weight':J_E, 
                'delay':delay}) 

    nest.Connect(noise,neurons)
    nest.Connect(neurons,spike_detector)
    
    # Create SO connections
    for pre_neuron in np.unique(senders_pre):
        times_pre_neuron = times_pre[senders_pre==pre_neuron]
        for bouton in np.arange(n_boutons_so):
            synapses_per_bouton = n_synapses_so('so')
            active_zones = nest.Create('parrot_neuron',synapses_per_bouton)
            spikes_at_active_zone = nest.Create('spike_generator',synapses_per_bouton)
            release_probability = pr_function()
            #synapse_type = select_synapse_type(stp)
            if release_probability>=pr_cc:
                synapse_type = 'STD'
            else:
                synapse_type = 'STF'
            for az in spikes_at_active_zone:
                spk_times = np.random.choice(times_pre_neuron,size=int(release_probability*len(times_pre_neuron)),replace=False)
                idx = np.argsort(spk_times)
                nest.SetStatus([az],{'spike_times':spk_times[idx]})
            nest.Connect(spikes_at_active_zone,active_zones,'one_to_one')
            nest.Connect(active_zones,neurons,
                         {'rule' : 'fixed_outdegree', 
                          'outdegree' : 1},
                           synapse_type)
                           
    # Create SR connections
    for pre_neuron in np.unique(senders_pre):
        times_pre_neuron = times_pre[senders_pre==pre_neuron]
        for bouton in np.arange(n_boutons_sr):
            synapses_per_bouton = n_synapses_sr('sr')
            active_zones = nest.Create('parrot_neuron',synapses_per_bouton)
            spikes_at_active_zone = nest.Create('spike_generator',synapses_per_bouton)
            release_probability = pr_function()
            #synapse_type = select_synapse_type(stp)
            if release_probability>=pr_cc:
                synapse_type = 'STD'
            else:
                synapse_type = 'STF'
            for az in spikes_at_active_zone:
                spk_times = np.random.choice(times_pre_neuron,size=int(release_probability*len(times_pre_neuron)),replace=False)
                idx = np.argsort(spk_times)
                nest.SetStatus([az],{'spike_times':spk_times[idx]})
            nest.Connect(spikes_at_active_zone,active_zones,'one_to_one')
            nest.Connect(active_zones,neurons,
                         {'rule' : 'fixed_outdegree', 
                          'outdegree' : 1},
                           synapse_type)
            
    
    # Simulate ------------------------------------------------------
    nest.Simulate(simtime)
    
    # Analyze data --------------------------------------------------
    events = nest.GetStatus(spike_detector,'events')[0]
    times = events['times']
    senders = events['senders']

    cc = []
    for neuron1 in neurons:
        spk_train1 = np.histogram(times[senders==neuron1],bins=time_bins)[0]
        for neuron2 in np.arange(neuron1+1,len(neurons)+1):
            spk_train2 = np.histogram(times[senders==neuron2],bins=time_bins)[0]
            cc.append(np.corrcoef(spk_train1,spk_train2)[0,1])
    cc = np.array(cc)
    
    return np.mean(cc[~np.isnan(cc)])

def main(master_seed,mult,stp,pr):
    pre_events = simulate_network_pre(master_seed)
    master_seed += 2
    cc_ssb = simulate_network_post(master_seed,pre_events,mult=mult,stp=stp,pr=pr,sb_type_so='SSB',sb_type_sr='SSB')
    master_seed += 2
    cc_ssb_so = simulate_network_post(master_seed,pre_events,mult=mult,stp=stp,pr=pr,sb_type_so='SSB',sb_type_sr='MSB')
    master_seed += 2
    cc_ssb_sr = simulate_network_post(master_seed,pre_events,mult=mult,stp=stp,pr=pr,sb_type_so='MSB',sb_type_sr='SSB')
    master_seed += 2
    cc_msb = simulate_network_post(master_seed,pre_events,mult=mult,stp=stp,pr=pr,sb_type_so='MSB',sb_type_sr='MSB')
    return (cc_ssb,cc_ssb_so,cc_ssb_sr,cc_msb),master_seed

master_seed = 7500
pr_cc = 0.5

                     # mult, stp, pr
all_cases = np.array([[True,True,True],
                      [False,True,True]])


for multiplicative,short_term_plasticity,release_probability in zip(all_cases[:,0],all_cases[:,1],all_cases[:,2]):
    cc_ssb = []
    cc_ssb_so = []
    cc_ssb_sr = []
    cc_msb = []
    for ii in np.arange(n_simulations):
        cc,master_seed = main(master_seed,mult=multiplicative,stp=short_term_plasticity,pr=release_probability)
        cc_ssb.append(cc[0])
        cc_ssb_so.append(cc[1])
        cc_ssb_sr.append(cc[2])
        cc_msb.append(cc[3])
        master_seed += 2
        
    extension = '_mult'*multiplicative+'_stp'*short_term_plasticity+'_pr'*release_probability+'_corr'+str(pr_cc)

    np.save("cc_ssb"+extension+".npy",cc_ssb)
    np.save("cc_ssb_so"+extension+".npy",cc_ssb_so)
    np.save("cc_ssb_sr"+extension+".npy",cc_ssb_sr)
    np.save("cc_msb"+extension+".npy",cc_msb)
