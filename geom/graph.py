# graph.py
from collections import defaultdict

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes              # id -> (x,y)
        self.edges = list(edges)        # list[(u,v)]
        self.adj   = defaultdict(list)  # id -> [edge_id]
        for eid,(u,v) in enumerate(self.edges):
            self.adj[u].append(eid)
            self.adj[v].append(eid)

    def degree(self, nid):
        return len(self.adj[nid])

    def edge_nodes(self, eid):
        return self.edges[eid]
    
    def add_node(self, xy):
        nid = len(self.nodes)
        self.nodes.append(tuple(xy))
        if nid not in self.adj:
            self.adj[nid] = []
        return nid

    def add_edge(self, u, v):
        eid = len(self.edges)
        self.edges.append((u, v))
        self.adj[u].append(eid)
        self.adj[v].append(eid)
        return eid

    def remove_edge(self, eid):
        u, v = self.edges[eid]
        self.edges[eid] = (-1, -1)
        if eid in self.adj.get(u, []): self.adj[u].remove(eid)
        if eid in self.adj.get(v, []): self.adj[v].remove(eid)

    def split_edge(self, eid, xy):
        """Разрезает ребро eid точкой xy. Возвращает (nid_new, eid1, eid2)."""
        u, v = self.edges[eid]
        assert u != -1 and v != -1, "edge removed"
        nid = self.add_node(xy)
        self.remove_edge(eid)
        e1 = self.add_edge(u, nid)
        e2 = self.add_edge(nid, v)
        return nid, e1, e2

    def clone(self):
        import copy
        return copy.deepcopy(self)