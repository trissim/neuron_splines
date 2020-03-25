import numpy as np
import matplotlib.pyplot as plt
def im_fig(figs,size=10,cmap='gray',show=True):
    if not hasattr(figs,'len'):
        figs = np.array(figs)
    dims = tuple(np.array(figs).shape)[:-2]
    if len(dims) == 1:
        rows = 1
        cols = dims[0]
    else:
        cols, rows = dims
    if rows > 1 :
        fig_list = []
        for row in range(rows): fig_list+=list(figs[row])
    else:
        fig_list = figs
    f, axarr = plt.subplots(rows,cols,figsize = (size,size))
    if hasattr(axarr,'shape'):
        ax_list = axarr.flatten()
    else:
        ax_list = [axarr]
    for i,ax in enumerate(ax_list):
        ax.imshow(fig_list[i],cmap=cmap)
    if not show:
        plt.close()
    plt.tight_layout()
    return (f, axarr)
