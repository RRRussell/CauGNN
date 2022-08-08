import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_groups = 4

hstgnn = (0.0173, 0.0240, 0.0333, 0.0441)
type1 = (0.0175, 0.0245, 0.0362, 0.0449)
type2 = (0.0175, 0.0245, 0.0362, 0.0449)
type3 = (0.0175, 0.0245, 0.0362, 0.0449)
type4 = (0.0175, 0.0245, 0.0362, 0.0449)

# means_women = (25, 32, 34, 20, 25)
# std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.1

opacity = 0.4
error_config = {'ecolor': '0.3'}

plt.ylim((0.015,0.055))

# rects1 = ax.bar(index, tegnn, bar_width,
#                 alpha=opacity, color='b',
#                 yerr=std_men, error_kw=error_config,
#                 label='Men')
rects1 = ax.bar(index - 2*bar_width, hstgnn, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='HSTGNN')
rects2 = ax.bar(index - bar_width, type1, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='type1')
rects3 = ax.bar(index, type2, bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='type2')
rects4 = ax.bar(index + bar_width, type3, bar_width,
                alpha=opacity, color='y',
                error_kw=error_config,
                label='type3')
rects5 = ax.bar(index + 2*bar_width, type4, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='type4')

ax.set_xlabel('horizon')
ax.set_ylabel('RAE')
ax.set_title('Performance of Variants of HSTGNN')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('3', '6', '12', '24'))
ax.legend()

fig.tight_layout()
plt.show()