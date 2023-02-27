import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

plt.rcParams.update({'font.size': 8})

cc_ssb_all = np.load("cc_ssb_mult_stp_pr_corr0.5.npy")
cc_ssb_so_all = np.load("cc_ssb_so_mult_stp_pr_corr0.5.npy")
cc_ssb_sr_all = np.load("cc_ssb_sr_mult_stp_pr_corr0.5.npy")
cc_msb_all = np.load("cc_msb_mult_stp_pr_corr0.5.npy")

figure = plt.figure(1,figsize=(4,2))

ax1 = figure.add_axes([0.15,0.2,0.3,0.7])
ax2 = figure.add_axes([0.65,0.2,0.3,0.7])

def axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_axes(ax,cc_ssb,cc_ssb_so,cc_ssb_sr,cc_msb):
    #ax.bar([1,2],[np.mean(cc_ssb),np.mean(cc_msb)],yerr=[np.std(cc_ssb),np.std(cc_msb)],edgecolor=['grey','k'],color='white')
    #ax.boxplot([cc_ssb,cc_msb])
    
    parts = ax.violinplot([cc_ssb,cc_ssb_so,cc_ssb_sr,cc_msb],showmeans=True,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor('black')
    #parts['cbars'].set_color('black')
    parts['cmeans'].set_color('black')
    #parts['cmins'].set_color('black')
    #parts['cmaxes'].set_color('black')
    
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['','',''])
    axes(ax)

plot_axes(ax2,cc_ssb_all,cc_ssb_so_all,cc_ssb_sr_all,cc_msb_all)
ax2.set_xticklabels(['SSB','SO SSB\nSR MSB','MSB'])
ax2.set_ylabel(r'$\mu_{cc}$')

shape = 2
scale = 0.15
x_gamma = np.linspace(0,1,1000)
gamma_distribution = gamma.pdf(x_gamma,shape,scale=scale)
ax1.plot(x_gamma,gamma_distribution,color='k')
ax1.set_xlabel(r"Relase probability $p_i$")
ax1.set_ylabel("pdf")


figure.text(0.02,0.95,'A',fontweight='bold')
figure.text(0.5,0.95,'B',fontweight='bold')

plt.savefig('supp_figure.svg',figsize=(4,2))
plt.show()
