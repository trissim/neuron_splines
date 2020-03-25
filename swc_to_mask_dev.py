import cv2
import pandas as pd
import navis
import imageio
import numpy as np
from matplotlib import pyplot as plt
from navis.interfaces import neuromorpho as nm
def get_edges(neuron):
  dgraph = navis.neuron2nx(neuron)
  segments = []
  for edge in dgraph.edges:
    segments.append((neuron.nodes.loc[edge[0]-1],neuron.nodes.loc[edge[1]-1]))
  return segments

def getPerpCoord(p1, p2):
   aX, aY = p1['x'],p1['y']
   bX, bY = p2['x'],p2['y']
   length = p2['radius']
   vX = bX-aX
   vY = bY-aY
   mag = np.sqrt(vX*vX + vY*vY)
   vX = vX / mag
   vY = vY / mag
   temp = vX
   vX = 0-vY
   vY = temp
   cX = bX + vX * length
   cY = bY + vY * length
   dX = bX - vX * length
   dY = bY - vY * length
   return (np.array([cX, cY])), (np.array([dX, dY]))

def segment_to_polygon(segment):
  connect1, connect2 = segment  
  p1 = getPerpCoord(connect2,connect1)
  p1_pX, p1_pY = p1[0], p1[1]
  p2 = getPerpCoord(connect1,connect2)
  p2_pX, p2_pY = p2[0], p2[1]
  return np.array([p1_pX, p1_pY, p2_pX, p2_pY])

def bbox_dimensions(neuron):
  height = np.abs(neuron.bbox[1][0] - neuron.bbox[1][1])
  width = np.abs(neuron.bbox[0][0] - neuron.bbox[0][1])
  origin = np.array([neuron.bbox[0][0] , neuron.bbox[1][0]])
  return origin, np.array([width,height])

def scale_to_image(point, neuron_space, img_dim):
  origin_point, size = neuron_space[0], neuron_space[1]
  rel_x = (point[0] - origin_point[0])/size[0]
  rel_y = (point[1] - origin_point[1])/size[1]
  scaled_x = rel_x*img_dim[0]
  scaled_y = rel_y*img_dim[1]
  return np.uint32([scaled_x,scaled_y])

def get_image_dimension(neuron,pixel_per_segment):
  origin, bbox_size = bbox_dimensions(neuron)
  scale_factor = pixel_per_segment/neuron.sampling_resolution 
  return np.uint32(scale_factor*bbox_size)
  
def neuron_to_img(neuron, absolute_size=None, min_width = None, min_height = None, pixel_per_segment = None, 
                  draw_segment = True, draw_node = True, node_rad = None, seg_rad = None, 
                  pad_percent = 0, node_color = 255, seg_color = 127):
  segments = get_edges(neuron)
  abs_coords = False
  if not absolute_size is None:
    img_dim = np.uint16((absolute_size[1],absolute_size[0]))
    img = np.zeros((img_dim[1],img_dim[0],1), np.uint16)
    abs_coords = True
  else:
      origin, bbox_size = bbox_dimensions(neuron)
      if not min_width is None:
        img_dim = np.uint16((min_width/bbox_size[0])*bbox_size)
      elif not min_height is None:
        img_dim = np.uint16((min_height/bbox_size[0])*bbox_size)
      elif not pixel_per_segment is None:
        img_dim = get_image_dimension(neuron, pixel_per_segment)
      else:
        img_dim = np.uint16(bbox_size)
      img = np.zeros((img_dim[1],img_dim[0],1), np.uint16)
      if not pad_percent == 0:
        y_pad = np.int16(img.shape[0]*(pad_percent/100.0))
        x_pad = np.int16(img.shape[1]*(pad_percent/100.0))
        img = np.squeeze(img)
        img = np.pad(img,((x_pad,x_pad),(y_pad,y_pad)),'constant')
        img = img[:,:,np.newaxis]
      else:
        y_pad = 0 
        x_pad = 0
      if not seg_rad is None:
        seg_rad = int(seg_rad*(img_dim[0]/bbox_size[0]) )
        if seg_rad == 0:
          seg_rad = 1
        scaled_seg_rad = int(seg_rad*(img_dim[0]/bbox_size[0]) )
  for segment in segments:
    if draw_segment:
      if seg_rad is None:
        polygon_points = segment_to_polygon(segment)
        if not abs_coords:
          points = []
          for point in polygon_points:
            scaled_point = scale_to_image(point,(origin,bbox_size),img_dim)
            scaled_point += np.uint16(np.array([y_pad,x_pad]))
            points.append(scaled_points)
        else:
          points = polygon_points
        points = np.int32(np.array(points))
        points = points.reshape(-1,1,2)
        img = cv2.fillConvexPoly(img, points, seg_color)
      else:
        points = []
        for node in segment:
          point = (np.uint16(node['x']),np.uint16(node['y']))
          if not abs_coords:
             scaled_point = scale_to_image(point,(origin,bbox_size),img_dim)
             scaled_point += np.uint16(np.array([y_pad,x_pad]))
             point = (scaled_point[0],scaled_point[1])
          points.append(point)
        img = cv2.line(img,points[0],points[1],seg_color,seg_rad)
  if draw_node:
    for index, node in neuron.nodes.iterrows():
      point = (np.uint16(node['x']),np.uint16(node['y']))
      if not abs_coords:
        scaled_node_point = scale_to_image(point,(origin,bbox_size),img_dim)
        scaled_node_point += np.uint16(np.array([y_pad,x_pad]))
        point = tuple(scaled_node_point)
      if abs_coords:
        radius = np.int32(node['radius'])
      else:
        if not node_rad is None:
          radius = np.int32(node_rad*img_dim[0]/bbox_size[0])
        else:
          radius = np.int32(node['radius']*img_dim[0]/bbox_size[0])
      img = cv2.circle(img, point, radius, node_color, -1)
  return img

def plt_nmid(nmid,res=5,draw_segment=True,draw_node=True):
  n = nm.get_neuron(nmid)
  n_img = neuron_to_img(n,res,draw_segment=draw_segment,draw_node=draw_node)
  plt.imshow(n_img)
  n.plot2d()

def node_by_id(neuron, id):
  index = node_ind_by_id(neuron,id)
  node = neuron.nodes.loc[index]
  return node

def dist_two_nodes(neuron,n1,n2):
  out1 = neuron.nodes.loc[n1]
  out2 = neuron.nodes.loc[n2]
  x = out1['x']-out2['x']
  y = out1['y']-out2['y']
  z = out1['z']-out2['z']
  length = np.sqrt(np.square(x)+np.square(y)+np.square(z))
  return length

def delete_node(neuron,shortest):
  middle_id = shortest[0]
  start_id = neuron.nodes.at[middle_id,'parent_id']
  neuron.nodes.drop(middle_id,inplace=True)
  end_id = shortest[2][1-shortest[2].index(start_id)]
  neuron.nodes.at[end_id,'parent_id'] = start_id
  return neuron

def prune_by_length(in_neuron,length):
  removed_seg = True
  neuron = in_neuron.copy()
  while(removed_seg):
    neuron.nodes.set_index('node_id',inplace=True,drop=False)
    adj_list = navis.neuron2nx(neuron).to_undirected()
    shortests = {}
    for adj in adj_list.adjacency():
      if len(adj[1]) == 2:
        middle = adj[0] 
        if not middle in shortests.keys():
          connectors = list(adj[1].items())
          total_length = connectors[0][1]['weight'] + connectors[1][1]['weight']
          shortest = (middle,total_length,(connectors[0][0],connectors[1][0]))
          shortests[middle]=shortest
    shortests = list(shortests.values())
    shortests.sort(key=lambda x: x[1],reverse=False)
    removed_seg = False
    for shortest in shortests:
      out1 = shortest[2][0]
      out2 = shortest[2][1]
      prune_length = dist_two_nodes(neuron, out1,out2)

      ratio_conserved = prune_length/shortest[1]
      bbox_size = bbox_dimensions(neuron)
      max_len = np.minimum(bbox_size[0],bbox_size[1])/10.0
      
      if prune_length/shortest[1] > 0.98:
        neuron = delete_node(neuron, shortest)
        removed_seg = True
        break
  indexes = list(neuron.nodes['node_id'])
  indexes = pd.Series([ind-1 for ind in indexes])
  neuron.nodes.set_index(indexes,inplace=True)
  return neuron

def compare_skeleton(neuron):
  clean = prune_by_length(neuron,5)
  n_img = neuron_to_img(neuron,min_width=512)
  clean_img = neuron_to_img(clean,node_rad=2, seg_rad=1,min_width=512)
  plt.imshow(clean_img)
  plt.imshow(n_img)
