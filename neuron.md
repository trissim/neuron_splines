---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import navis 
import numpy as np
import scipy
import utils
import ims
import swc_to_mask as nu
from scipy import interpolate
```

```python
###############
    #UTILS
###############

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from simplification.cutil import *

### Basic functions for swc <-> segments <-> points conversions
get_node = lambda neuron, node_id : neuron.nodes.loc[neuron.nodes['node_id'] == node_id] 
get_node_pos = lambda node : (np.float(node['x']),np.float(node['y']))
get_point_list = lambda segment : [get_node_pos(get_node(segment,node_id)) for node_id in segment.nodes['node_id']]
get_segments = lambda neuron : [navis.subset_neuron(neuron,list(n)) for n in neuron.segments]
get_seg_as_pl = lambda segment : [(a,-1.0*b) for (a,b) in get_point_list(segment)]



def bspline(cv, n=100, degree=3):
    """ Calculate n samples on a bspline
        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
    """
    # taken from 
    # https://stackoverflow.com/questions/28279060/splines-with-python-using-control-knots-and-endpoints

    cv = np.asarray(cv)
    count = cv.shape[0]
    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree,dtype='int')
    # Calculate query range
    u = np.linspace(0,(count-degree),n)

    # Calculate result
    return np.array(interpolate.splev(u, (kv,cv.T,degree))).T

def plot_segments(lines,color='g'):
    default_opts = { 'name' : None,
                     'plttype' : plt.plot,
                     'color' : None }
    
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    #pre processing of inputs
    for i,line in enumerate(lines):
        #make sure its a dict
        if not type(line) is dict:
           print('converting')
           line = {'points' : line,
                     'name' : "line " + str(i)}
        print("points", type(line['points'][0]))
        # make sure points are a 2d np array 
        if type(line['points'][0]) is tuple:
            line['points'] = np.array([list(pair) for pair in line['points']])
        # fill in missing params with defaults
        for k,v in default_opts.items():
            if not k in line.keys():
                if v is None:
                    line[k] = colors[i] if v == 'color' else i
                else:
                    line[k] = v
        lines[i] = line
    for i,line in enumerate(lines):
        x,y = lines[i]['points'].T
        lines[i]['plttype'](x,y,label='%s'%line['name'],color=colors[line['color']])
    plt.minorticks_on()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#Fit curve from points
import curve_fit_nd
def curve_fit(s, error, corner_angle, is_cyclic):
    if corner_angle is None:
        corner_angle = 10
    #if type(s) is list:
        #s = tuple(s) 
    c = curve_fit_nd.curve_from_points(s, error, corner_angle, is_cyclic)
    return c

def spline_points(pts):
    knots = []
    for i in pts:
        knots = knots + list(i[1])
    print(knots)
    return knots 


```

```python
########################
    #TEST PLOTS
#########################

def plot_spline_pts(spline):
    pts = {'points' : spline_points(spline),
             'name' : 'spline ctrl pts' ,
            'color' : 'c'}
    plot_segments([pts])
    
def plot_neuron(segments):
    interp_segs = []
    for i,segment in enumerate(segments):
        seg_pts = get_seg_as_pl(segment)
        spline = curve_fit(seg_pts, 1, 0, False)
        spline_pt_list = spline_points(spline)
        interp = bspline(spline_pt_list)
        interp = {'points' : interp,
                  'name' : 'interpolated splines ' + str(i),
                  'color' : 'c'}
        interp_segs.append(interp)
    plot_segments(interp_segs)


```

```python
########################
    #LOAD FILES
#########################

ims_file = '/home/levodextro/sd/School/Data/Microscopy/mardja_fc/FC-A Imaris Analysis/FC-A vs FC-7 plate 2_C01_s4_w1F6C80250-C491-4477-8F05-8FF7BC257943.ims'
neuron = navis.from_swc('/home/levodextro/sd/School/Data/Microscopy/mardja_fc/NLMorphologyConverter/cnicswc/FC-A vs FC-7 plate 2_C01_s4_w1F6C80250-C491-4477-8F05-8FF7BC257943.swc')
raw_img = ims.ims(ims_file).img
```

```python
########################
    #TESTS
#########################

utils.im_fig([raw_img])
segments = get_segments(neuron)
seg_pts = get_seg_as_pl(segments[6])
spline = curve_fit(seg_pts, 1, 0, False)
interp = bspline(spline_pt_list)
interp = {'points' : interp,
          'name' : 'interpolated splines'}
spline_pt_list = {'points' : spline_pt_list,
          'name' : 'spline_pt_list',
          'plttype' : plt.scatter}
seg_pts_raw = {'points' : seg_pts,
          'name' : 'seg_pts_raw'}
plot_segments([interp,spline_pt_list,seg_pts_raw])

```

```python
########################
    #SANDBOX
#########################
for segment in segments:
    seg_pts = get_seg_as_pl(segments[6])
#plot_spline_pts(spline)
```

```python
########################
    #TESTING GROUNDS
#########################

utils.im_fig([raw_img])
segments = get_segments(neuron)
seg_pts = get_seg_as_pl(segments[6])
print(seg_pts)

spline = curve_fit(seg_pts, 1, 0, False)
print(spline)
#print(spline_pt_list)
#print(interp)
spline_pt_list = spline_points(spline)
interp = bspline(spline_pt_list)
interp = {'points' : interp,
          'name' : 'interpolated splines'}
spline_pt_list = {'points' : spline_pt_list,
          'name' : 'spline_pt_list',
          'plttype' : plt.scatter}
seg_pts_raw = {'points' : seg_pts,
          'name' : 'seg_pts_raw'}
plot_segments([interp,spline_pt_list,seg_pts_raw])
```


```python
pt_list = [(a,-1.0*b) for (a,b) in get_point_list(segment)]
spline = curve_fit(pt_list, 1, 10, False)
print(spline[0])

def plot_curve(interp_B):
    vis_config = VisMPL.VisConfig(legend=False, axes=False, figure_dpi=120)
    vis_obj = VisMPL.VisCurve2D(vis_config)
    interp_B.vis = vis_obj
    interp_B.render()

def get_spline(segment,tol=2,order=3,topology=True):
    pt_list = [(a,-1.0*b) for (a,b) in get_point_list(segment)]
    #seg[seg_num].plot2d()
    raw_pt_num = pt_list 
    if topology:
        simple_pt_list = simplify_coords_vwp(pt_list,tol)
    else:
        simple_pt_list = simplify_coords(pt_list,tol)
    interp_B = interpolate_curve(simple_pt_list,order)
    interp_B = interpolate_curve(simple_pt_list,order)
    return interp_B

def neuron_to_splines(neuron,tol=2,order=3):
    return [get_spline(segment,tol=tol,order=order) for segment in get_segments(neuron)]
    
    
```

```python
from geomdl.fitting import interpolate_curve
from geomdl.fitting import approximate_curve
from geomdl.visualization import VisMPL
ims_file = '/home/levodextro/sd/School/Data/Microscopy/mardja_fc/FC-A Imaris Analysis/FC-A vs FC-7 plate 2_C01_s4_w1F6C80250-C491-4477-8F05-8FF7BC257943.ims'
neuron = navis.from_swc('/home/levodextro/sd/School/Data/Microscopy/mardja_fc/NLMorphologyConverter/cnicswc/FC-A vs FC-7 plate 2_C01_s4_w1F6C80250-C491-4477-8F05-8FF7BC257943.swc')
from importlib import reload
reload(ims)
raw_img = ims.ims(ims_file).img
#raw_img = nu.neuron_to_img(neuron,min_width=500)
utils.im_fig([raw_img])
segments = get_segments(neuron)
#plot_curve(get_spline(segments[3]))
neuron.plot2d()
for segment in get_segments(neuron):
    segment.plot2d()
    plot_curve(get_spline(segment,tol=1,order = 3))
    

```

```python
neuron = navis.from_swc('/home/levodextro/sd/School/Data/Microscopy/mardja_fc/NLMorphologyConverter/cnicswc/FC-A vs FC-7 plate 2_C01_s4_w1F6C80250-C491-4477-8F05-8FF7BC257943.swc')
segments = get_segments(neuron)
for segment in get_segments(neuron):
    segment.plot2d()
    plot_curve(get_spline(segment,tol=5,order = 3))
```


```python
neuron = navis.from_swc('/home/levodextro/sd/School/Data/Microscopy/mardja_fc/NLMorphologyConverter/cnicswc/FC-A vs FC-7 plate 2_C01_s4_w1F6C80250-C491-4477-8F05-8FF7BC257943.swc')
segments = get_segments(neuron)
for segment in get_segments(neuron):
    segment.plot2d()
    plot_curve(get_spline(segment,tol=1,topology=False,order = 3))
```


```python

neuron = navis.from_swc('/home/levodextro/sd/School/Data/Microscopy/mardja_fc/NLMorphologyConverter/cnicswc/FC-A vs FC-7 plate 2_C01_s4_w1F6C80250-C491-4477-8F05-8FF7BC257943.swc')
segments = get_segments(neuron)
for segment in get_segments(neuron):
    segment.plot2d()
    plot_curve(get_spline(segment,tol=5,topology=False,order = 3))
```


```python

```
