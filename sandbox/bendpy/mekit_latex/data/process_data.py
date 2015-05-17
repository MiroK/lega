# from matplotlib import rc
# rc('text', usetex=True) 
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt
import numpy as np


what = 'shen'
data1 = np.loadtxt('s_one_up_down_%s_1' % what)
data2 = np.loadtxt('s_one_up_down_%s_2' % what)

# Plot condition numers
if True:
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data1[:, 0], data1[:, 4], label=r'$\kappa_1$',
            marker='x', linestyle='None')
    ax.plot(data2[:, 0], data2[:, 4], label=r'$\kappa_2$',
            marker='o', linestyle='None', ms=10, fillstyle='none')
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$\kappa$')
    plt.legend(loc='lower right')
    plt.savefig('Precond_%s_cond.pdf' % what)

if True:
    # Plot lmin, lmax. twin axes
    fig, ax1 = plt.subplots()
    l0, = ax1.plot(data1[:, 0], data1[:, 2], label='min_1', color='b',
            marker='x', linestyle='None')
    l1, = ax1.plot(data2[:, 0], data2[:, 2], label='min_2', color='b',
            marker='o', linestyle='None', ms=10, fillstyle='none')
    ax1.set_xlabel('$n$')
    ax1.set_ylabel('$\lambda_{min}$', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    #
    ax2 = ax1.twinx()
    l2, = ax2.plot(data1[:, 0], data1[:, 3], label='max_1', color='r',
            marker='x', linestyle='None')
    l3, = ax2.plot(data2[:, 0], data2[:, 3], label='max_2', color='r',
            marker='o', linestyle='None', ms=10, fillstyle='none')
    ax2.set_ylabel('$\lambda_{max}$', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    fig.legend([l0, l1, l2, l3], [r'$\lambda_{min, 1}$', r'$\lambda_{min, 2}$',
    r'$\lambda_{max, 1}$', r'$\lambda_{max, 2}$'], bbox_to_anchor=(0.90, 0.50))

    plt.savefig('prec_%s_spectrum.pdf' % what)
#
plt.show()
