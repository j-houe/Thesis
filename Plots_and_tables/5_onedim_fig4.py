import numpy as np
import matplotlib.pyplot as plt


amax = 0.03
a0 = amax / 10
amin = amax / 10000


def annealing_cos(start, end, period_len):
    pct = np.linspace(0, 1, period_len)
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


xs = np.arange(0, 1, 0.01)
ys = np.concatenate((annealing_cos(a0, amax, 20), annealing_cos(amax, amin, 80)))
plt.plot(xs, ys, color='dodgerblue', lw=3)
plt.xlabel('Iterations', size=20)
plt.xticks([], [])
plt.ylabel('Learning Rate', size=20)
plt.yticks([amax, a0, amin],[r'$\alpha_{max}$', r'$\alpha_0$', r'$\alpha_{min}$'], size=20)
plt.title('1cycle Learning Rate Policy', size=20)
plt.gcf().set_size_inches(8, 4)
plt.tight_layout()
plt.show()
