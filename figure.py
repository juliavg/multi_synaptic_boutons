import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 7})

cc_ssb_all = np.load("cc_ssb_mult_stp_pr.npy")
cc_ssb_so_all = np.load("cc_ssb_so_mult_stp_pr.npy")
cc_ssb_sr_all = np.load("cc_ssb_sr_mult_stp_pr.npy")
cc_msb_all = np.load("cc_msb_mult_stp_pr.npy")
cc_ssb_multiplicative = np.load("cc_ssb_mult.npy")
cc_ssb_so_multiplicative = np.load("cc_ssb_so_mult.npy")
cc_ssb_sr_multiplicative = np.load("cc_ssb_sr_mult.npy")
cc_msb_multiplicative = np.load("cc_msb_mult.npy")
cc_ssb_failure = np.load("cc_ssb_pr.npy")
cc_ssb_so_failure = np.load("cc_ssb_so_pr.npy")
cc_ssb_sr_failure = np.load("cc_ssb_sr_pr.npy")
cc_msb_failure = np.load("cc_msb_pr.npy")
cc_ssb_STP = np.load("cc_ssb_stp.npy")
cc_ssb_so_STP = np.load("cc_ssb_so_stp.npy")
cc_ssb_sr_STP = np.load("cc_ssb_sr_stp.npy")
cc_msb_STP = np.load("cc_msb_stp.npy")

figure = plt.figure(1,figsize=(3,3))

ax1 = figure.add_axes([0.6,0.55,0.35,0.4])
ax2 = figure.add_axes([0.2,0.05,0.15,0.25])
ax3 = figure.add_axes([0.5,0.05,0.15,0.25])
ax4 = figure.add_axes([0.8,0.05,0.15,0.25])

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
    
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['','','',''])
    axes(ax)

plot_axes(ax1,cc_ssb_all,cc_ssb_sr_all,cc_ssb_so_all,cc_msb_all)
plot_axes(ax2,cc_ssb_multiplicative,cc_ssb_sr_multiplicative,cc_ssb_so_multiplicative,cc_msb_multiplicative)
plot_axes(ax3,cc_ssb_failure,cc_ssb_sr_failure,cc_ssb_so_failure,cc_msb_failure)
plot_axes(ax4,cc_ssb_STP,cc_ssb_sr_STP,cc_ssb_so_STP,cc_msb_STP)

ax1.set_xticklabels(['SSB','SO MSB\nSR SSB','SO SSB\nSR MSB','MSB'],rotation='vertical')
ax1.set_ylabel(r'$\mu_{cc}$')
ax2.set_ylabel(r'$\mu_{cc}$')

figure.text(0.01,0.95,'A',fontweight='bold')
figure.text(0.4,0.95,'B',fontweight='bold')
figure.text(0.05,0.32,'C',fontweight='bold')
figure.text(0.4,0.32,'D',fontweight='bold')
figure.text(0.7,0.32,'E',fontweight='bold')


plt.savefig('figure.svg',figsize=(3,3))
plt.show()
