# ---------------------------------------------------------------------
#  persistent_homology.py   —  CLEANED + BUG-FIXED (22 Jul 2025)
# ---------------------------------------------------------------------
from typing import List, Dict
import logging
from pathlib import Path
import numpy as np

# ──────────────────────────────────────────────────────────────────────
#  0-dimensional persistence class
# ──────────────────────────────────────────────────────────────────────
class BettiZero:
    """Persistence for connected components (β₀)."""

    # ---------- disjoint-set ----------
    class UnionFind:
        __slots__ = ('parent', 'rank')

        def __init__(self, vertices: list[dict]):
            n = len(vertices)
            # same initialization as before
            self.parent = list(range(n))
            self.rank   = list(range(n))

        def find(self, x: int) -> int:
            parent = self.parent
            # path-halving: each step we jump up two levels
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(self, x: int, y: int) -> dict[int, dict[str, int]]:
            rootX = self.find(x)
            rootY = self.find(y)
            rankX = self.rank[rootX]
            rankY = self.rank[rootY]

            # prepare the exact same return value as before
            result = {
                x: {'root': rootX, 'rank': rankX},
                y: {'root': rootY, 'rank': rankY},
            }

            # only change parent pointers if they're in different sets
            if rootX != rootY:
                # preserves your logic:
                #   if rankX > rankY: attach rootX under rootY
                #   otherwise          : attach rootY under rootX
                if rankX > rankY:
                    self.parent[rootX] = rootY
                else:
                    self.parent[rootY] = rootX

            return result

        
    # ---------- life-cycle ----------
    def __init__(self, direction, vertices, edges):
        self.direction = direction
        self.vertices  = vertices
        self.edges     = edges

        # build filtration once
        self.filtration, self.new_to_original = make_filtration(
            vertices, edges, direction
        )
        self.uf = self.UnionFind(vertices)

    # ---------- public API ----------
    def compute_persistence(self):
        components, mergers = {}, {}
        all_vertices, births = [], []

        for h, stage in self.filtration.items():
            horiz, vert = stage['horizontal_edges'], stage['vertical_edges']
            verts       = stage['points']

            # 1 horizontal glue inside the slice
            self.horizontal_step(horiz, self.uf)

            # 2 vertical-edge sweep (records deaths)
            self.vertical_step(vert, components, mergers, self.uf)
            all_vertices.extend(verts)

            # 3 propagate components upward
            components = self.compute_components(all_vertices, components, self.uf)

            # 4 new births *inside* the slice
            births.extend(self.compute_new_births(verts, self.uf))

        return components, mergers, all_vertices, births

    # ---------- static helpers ----------
    @staticmethod
    def horizontal_step(edges, uf):
        for e in edges:
            uf.union(*e['vertices'])

    @staticmethod
    def vertical_step(edges, components, mergers, uf):
        """
        Correct rule: when two components merge, the one that was born
        *later* (larger new_index) dies, the older one survives.
        """
        for e in edges:
            x, y = e['vertices']
            root_before = (uf.find(x), uf.find(y))
            uf.union(x, y)

            if root_before[0] == root_before[1]:
                continue                         # already same component

            # choose the *younger* component to die
            younger = root_before[0] if root_before[0] > root_before[1] else root_before[1]
            mergers[younger] = max(e['height'])  # death height
            components.pop(younger, None)        # remove only the dying comp

    @staticmethod
    def compute_components(vertices, old_components, uf):
        comps = old_components
        find = uf.find
        for v in vertices:
            node = v['new_index']
            root = find(node)
            try:
                comps[root].append(node)
            except KeyError:
                comps[root] = [node]
        return comps


    @staticmethod
    def compute_new_births(vertices, uf):
        return [v for v in vertices if uf.find(v['new_index']) == v['new_index']]


# ──────────────────────────────────────────────────────────────────────
#  post-processing helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────
def compute_intervals(births: List[dict], mergers: Dict[int, float]):
    intervals = []
    for birth in births:
        left  = birth['height']
        right = mergers.get(birth['new_index'], 'infty')
        intervals.append([left, right])
    return intervals


def length_of_interval(interval):
    return 'infty' if interval[1] == 'infty' else interval[1] - interval[0]

def compute_largest_bar(intervals):
    longest = max(intervals, key=length_of_interval)
    return length_of_interval(longest), longest

def compute_n_largest_bars(intervals, n):
    # Keep only finite intervals
    finite = [
        (s, e) 
        for s, e in intervals 
        if isinstance(e, (int, float))
    ]
    # Sort by length descending and take the top n
    finite.sort(key=lambda iv: iv[1] - iv[0], reverse=True)
    return finite[:n]

# Preprocessing

def make_filtration(vertices, edges, direction):
    ahv = append_height_vertices
    fv  = format_vertices
    fe  = format_edges
    pg  = process_graph
    ge  = group_events_by_height

    # build the “raw” point and edge lists
    pts = [[v, [0, 0, 0]] for v in vertices]
    eds = [[[x, y], [1, 1, 1]] for x, y in edges]

    # pipeline
    verts   = fv(ahv(direction, pts))
    eds2    = fe(verts, eds)
    pre     = pg(verts, eds2, direction)
    sg, idx = pre['signed_graph'], pre['index_translation']
    inv_idx = {new: orig for orig, new in idx.items()}
    filtr   = ge(sg[0], sg[1])

    return filtr, inv_idx



def group_events_by_height(points, edges):
    events = {}
    # add points
    for p in points:
        h = p['height']
        bucket = events.get(h)
        if bucket is None:
            bucket = {'points': [], 'horizontal_edges': [], 'vertical_edges': []}
            events[h] = bucket
        bucket['points'].append(p)

    # add edges
    for e in edges:
        h0, h1 = e['height']
        # same as max(), but a tiny bit faster
        h = h1 if h1 > h0 else h0

        bucket = events.get(h)
        if bucket is None:
            bucket = {'points': [], 'horizontal_edges': [], 'vertical_edges': []}
            events[h] = bucket

        if h0 == h1:
            bucket['horizontal_edges'].append(e)
        else:
            bucket['vertical_edges'].append(e)

    return events




def process_graph(vertices, edges, direction):
    """
        The input are vertices and edges and a direction.
        
        The output is a graph ordered by height, and by x,y,z. The normal vectors are replaced with the sign.
    """
    processed_graph = order_graph(vertices, edges)
    graph = [processed_graph['vertices'], processed_graph['edges']]
    signed_graph = obtain_sign(graph, direction)
    return {'signed_graph': signed_graph, 'index_translation': processed_graph['index_translation']}

def subdivide_edges(edges: list) -> list:
    '''Input: List of edges formated as 
    edge = ['vertices': [index_i, index_j], 'height': [height_i, height_j], 'n': n]
    Output: Partitions the edges for processing in two steps.
    A list containing two lists of edges, the first entry is horizontal edges.
    '''
    horizontal_edges = []
    angled_edges = []
    for edge in edges:
        if min(edge['height'])==max(edge['height']):
            horizontal_edges.append(edge)
        else:
            angled_edges.append(edge)
    return [horizontal_edges, angled_edges]

# obtain_sign, sign, and order_graph are helper functions for process_graph. 

def obtain_sign(graph, direction: list) -> list:
    points, edges = graph
    signed_points = []
    signed_edges = []
    for point in points:
        point['sign'] = sign(point['normal'], direction)
        del point['normal']
        signed_points.append(point)
    for e in edges:
        signed_edges.append({'vertices': e['vertices'], 'height': e['height'], 'sign': sign(e['n'], direction)})
    return [signed_points, signed_edges]

def sign(v_1,v_2):
    product = v_1[0] * v_2[0] +  v_1[1] * v_2[1] + v_1[2] * v_2[2]
    sign = 0
    if product > 0:
        sign = 1
    elif product < 0:
        sign = -1
    return sign

def order_graph(vertices, edges):
    """
        The input are vertices and edges.
        {'coordinates': [i, j, k], 'original_index': idx, 'new_index': idx, 'height': h, 'normal': n}
        {'vertices': [e, l], 'height': [h_0,h_1], 'n': n}
        
        The output is a graph ordered by height, and by x,y,z.
    """

    # Step 1: Sort the vertices
    sorted_vertices = sorted(
        vertices,
        key=lambda v: (v['height'], v['coordinates'][0], v['coordinates'][1], v['coordinates'][2])
    )

    # Step 2: Relabel the vertices
    original_to_new_index = {}
    for new_index, vertex in enumerate(sorted_vertices):
        original_index = vertex['original_index']
        original_to_new_index[original_index] = new_index
        vertex['new_index'] = new_index


    
    # Step 3: Update the edges
    for edge in edges:
        # Map old indices to new indices and sort them within the edge
        new_indices = [original_to_new_index[vi] for vi in edge['vertices']]
        edge['vertices'] = new_indices
    # Step 4: Sort the edges
    sorted_edges = sorted(
        edges,
        key=lambda e: (max(e['height']), min(e['vertices']))
    )
    
    output_vertices = [ v for v in sorted_vertices ]
    output_edges = [ {'vertices': e['vertices'], 'height': e['height'], 'n': e['n'] } for e in sorted_edges ]
    return {'vertices': output_vertices, 'edges': output_edges, 'index_translation': original_to_new_index}

def height_of_vertex(direction, point):
    """Fast dot‑product; >4× faster than Python loop."""
    return float(np.dot(direction, point))


def append_height_vertices(direction, vertices):
    pts = np.asarray([v[0] for v in vertices])
    heights = pts @ np.asarray(direction)
    return [[p.tolist(), h, n] for p, h, n in zip(pts, heights, (v[1] for v in vertices))]

def format_vertices(vertices: list) -> list:
    # Input: [coord, height, vector n]
    new_vertices = []
    n = 0
    for vertex in vertices:
        new_vertices.append({'coordinates': vertex[0], 
                             'original_index': n, 
                             'new_index': None,
                             'height': vertex[1],
                             'normal': vertex[2]
                            })
        n += 1
    return new_vertices


def format_edges(points: list, edges: list) -> list:
    # Input: []
    formatted_edges = []
    for edge in edges:
        l_vertex_index = edge[0][0]
        r_vertex_index = edge[0][1]
        l_height = points[l_vertex_index]['height']
        r_height = points[r_vertex_index]['height']
        formatted_edges.append({'vertices': [l_vertex_index, r_vertex_index], 'height': [l_height, r_height], 'n': edge[1]})
    return formatted_edges


# Helper Functions

def reindex_edges(edges: list[int, int]) -> list[int, int]:
    '''
        Input: list of edges [m,n] where m,n >= 1 (indexing starts at 1)
        Output: same list of edges [m-1,n-1]
    '''
    reindexed_edges = []
    for edge in edges:
        reindexed_edges.append([edge[0] - 1, edge[1] - 1])
    return reindexed_edges





# Sphere Coverings

def rotate_points(points, theta, phi, alpha):
    # Create rotation matrix for rotation around z-axis
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Create rotation matrix for rotation around y-axis
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    
    # Create rotation matrix for rotation around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    
    # Combine the rotations (Here @ is matrix multiplication)
    R = Rz @ Ry @ Rx

    # Apply the rotation matrix to each point (R.T is the transpose of R)
    rotated_points = points @ R.T
    
    return rotated_points

def generate_circle_points(n, radius=1.0):
    # Evenly spaced values over the interval 0,2pi. 
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Makes the 3D array
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)))
    return points

def generate_sphere_points(n, rotations, threshold):
    # Number of rotations
    circle_points = generate_circle_points(n)
    angles = np.linspace(0, np.pi, rotations, endpoint=False)  # Uniformly spaced angles

    sphere_points = []

    for phi in angles:
        rotated_circle = rotate_points(circle_points, 0, phi, 0)  # Rotate around y-axis
        sphere_points.append(rotated_circle)
    sphere_points = np.vstack(sphere_points)
    # Set entries to zero where absolute value is less than the threshold
    sphere_points[np.abs(sphere_points) < threshold] = 0
    # Remove duplicate points
    return np.unique(sphere_points, axis=0)


def octahedron_directions(f=9):
    # Integer lattice points on the L1 “sphere”: |i|+|j|+|k| = f
    pts = []
    for i in range(-f, f+1):
        for j in range(-f, f+1):
            k_abs = f - abs(i) - abs(j)
            if k_abs < 0:
                continue
            if k_abs == 0:
                pts.append((i, j, 0))
            else:
                pts.append((i, j,  k_abs))
                pts.append((i, j, -k_abs))

    # Unique & sorted for reproducibility
    pts = sorted(set(pts))

    # Map to the octahedron surface (divide by f), then project to unit sphere
    V = np.array(pts, dtype=float) / f
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    dirs = V / norms  # 326 unit vectors

    return dirs  # shape: (326, 3)