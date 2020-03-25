import h5py
import numpy as np
import cv2
import networkx as nx
import navis

class ims:
    def __init__(self, path, has_img=True, has_tracing=True):
        self.path = path
        self.data = self.ims_data(self.path)
        if has_tracing:
            self.segments = self.get_segments(self.data)
            self.vertices = self.get_vertices(self.data)
            self.swc = self.swc_from_data(self.data)
        self.img = self.img_from_data(self.data)
    
    def ims_data(self,ims_path):
        return h5py.File(ims_path, 'r')
    
    def get_segments(self,ims_data):
            return np.array(ims_data['Scene']["Content"]['Filaments0']['Graphs']['Segments'])
        
    def get_vertices(self,ims_data):
            return np.array(ims_data['Scene']["Content"]['Filaments0']['Graphs']['Vertices'])
        
    def img_from_data(self,ims_data):
        img = np.array(ims_data['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data'])
        def get_last_2d(x):
            m,n = x.shape[-2:]
            return x.flat[:m*n].reshape(m,n)
        img = get_last_2d(img)
        mask = img == 0
        rows = np.flatnonzero((~mask).sum(axis=1))
        cols = np.flatnonzero((~mask).sum(axis=0))
        crop = img[rows.min():rows.max()+1, cols.min():cols.max()+1]
        return crop
    
    def img_from_path(self,ims_path):
        ims_file = self.ims_data(ims_path)
        return self.img(ims_file)
    
    def swc_from_data(self, ims_data):
        def get_head_ims(segments):
            heads = []
            for i,segment in enumerate(segments):
                if not segment[0] in list(segments[:,1]):
                    heads.append(segments[i][0])
            return np.unique(np.array(heads))
        gen_node_attr = lambda verts,col: {index:vert[col] for index,vert in enumerate(verts)}
        segments=np.array(ims_data['Scene']["Content"]['Filaments0']['Graphs']['Segments'])
        vertices=np.array(ims_data['Scene']["Content"]['Filaments0']['Graphs']['Vertices'])
        swc = nx.DiGraph()
        swc.add_edges_from(segments)
        swc.add_nodes_from(np.arange(len(vertices)))
        head = get_head_ims(segments)
        if len(head) == 1:
            head = head[0]
        swc = nx.dfs_tree(swc,head)
        attrs = ['x','y','z','radius','label']
        for i,attr in enumerate(attrs):
            attr_dict = gen_node_attr(vertices,i)
            nx.set_node_attributes(swc,attr_dict,attr)
        return navis.TreeNeuron(swc)
    
    def swc_from_path(self, ims_path):
        self.ims_data(ims_path)
        return self.swc_from_data(ims)
    
    def seg_from_nodes(self,radius=None):
        if radius is None:
            radius = lambda vert: vert[3]
        node_img = np.zeros(self.img.shape,np.uint8)
        for vert in self.vertices:
            coords = tuple(np.uint32(np.array(vert)[:-3]))
            node_img = cv2.circle(node_img, coords, radius(vert), 255, -1)
        return np.array(node_img)
    
