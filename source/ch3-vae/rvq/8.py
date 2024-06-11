import torch
from vector_quantize_pytorch import ResidualVQ
import numpy as np
import matplotlib.pyplot as plt
import os

n_dim = 256  # number of dimensions
cb_len = 256  # codebook length
K = 4  # number of codebooks
npoints_hd = 4096  # number of data points in high-dim space

d_choices = [2, 3, 6, 8, 16, 32, 64, 128, 256, 512]  # dimensions to try
cb_lengths = [25, 64, 256, 1024, 2048]  # codebook lengths
K_choices = [1, 2, 3, 4, 6, 8, 10]  # variable numbers of codebooks

results = torch.empty((len(d_choices), len(cb_lengths), len(K_choices))).cpu()
for q1, n_dim in enumerate(d_choices):
    for q2, cb_len in enumerate(cb_lengths):
        for q3, K in enumerate(K_choices):
            residual_vq = ResidualVQ(
                dim=n_dim,
                codebook_size=cb_len,
                num_quantizers=K,
                kmeans_init=True,  # set to True
                kmeans_iters=10  # number of kmeans iterations to calculate the centroids for the codebook on init
            )
            torch.manual_seed(0)
            x = torch.randn(1, npoints_hd, n_dim)
            quantized, indices, commit_loss = residual_vq(x)
            error = ((quantized - x) ** 2).mean()
            results[q1, q2, q3] = error

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='notebook'
import plotly.express as px
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.graph_objects as go

def plot_results(results, d_choices, cb_lengths, K_choices, log_xy=True, log_z=False, vary_K=False,
                camera=dict( up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=-0.2), eye=dict(x=1.3, y=1.3, z=1.25)),
                width=750,):
    y, ylabel = np.array(d_choices), 'd'
    if vary_K:
        x, xlabel = np.array(K_choices), 'K'
        z = results[:,-1,:]
    else:
        x, xlabel = np.array(cb_lengths), 'cb_len'
        z = results[:,:,3]
    z, zlabel = z.numpy(), 'error'

    if log_xy:
        if not vary_K:
            x, xlabel = np.log10(x), f"log10( {xlabel} )"
        y, ylabel = np.log10(y), f"log10( {ylabel} )"
    if log_z:
        z, zlabel = np.log10(z), f"log10( {zlabel} )"

    X, Y = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=z, )], )
    fig.update_layout(scene = dict(xaxis_title=xlabel,yaxis_title=ylabel,zaxis_title=zlabel))
    fig.update_layout(template='plotly_dark',
        autosize=False,
        width=width,
        height=400,
        margin=dict(l=20,r=20,b=10,t=20,pad=4),
        #paper_bgcolor="LightSteelBlue",
        scene_camera = camera,
    )
    return fig

fig = plot_results(results, d_choices, cb_lengths, K_choices, log_xy=True, log_z=False)
fig.write_html(f'images/8.html')
