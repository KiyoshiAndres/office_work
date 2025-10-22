from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, TypedDict

Vertex = int
Edge = Tuple[int, int]  # undirected, always stored with u < v
Face = Tuple[int, int, int]

def norm_edge(u: int, v: int) -> Edge:
    '''
    Given two integers, it returns an ordered edge.
    '''
    return (u, v) if u < v else (v, u)

def face_edges(f: Face) -> List[Edge]:
    '''
    Given a face, it returns its ordered edges in a list.
    '''
    a, b, c = f
    return [norm_edge(a, b), norm_edge(b, c), norm_edge(c, a)]

def build_primal_graph(V: List[Vertex], E: List[Edge]) -> Dict[Vertex, Set[Vertex]]:
    G = {v: set() for v in V}
    for u, v in E:
        u, v = (u, v) if u < v else (v, u)
        G[u].add(v)
        G[v].add(u)
    return G

def incident_faces_per_edge(F: List[Face]) -> Dict[Edge, List[int]]:
    inc: Dict[Edge, List[int]] = defaultdict(list)
    for fi, f in enumerate(F):
        for e in face_edges(f):
            inc[e].append(fi)
    return inc

def boundary_edges_and_components(V: List[Vertex],
                                  F: List[Face]) -> Tuple[Set[Edge], Dict[Edge, int], int]:
    """
    Identify boundary edges and group them by boundary component.
    Returns: boundary_edge_set, edge_to_component_id, number_of_boundary_components
    """
    inc = incident_faces_per_edge(F)
    boundary_edges = {e for e, faces in inc.items() if len(faces) == 1}

    # Build adjacency graph on boundary edges via shared vertices
    # Nodes are edges; connect two boundary edges if they share a vertex.
    # Each connected component corresponds to one boundary loop.
    adj: Dict[Edge, Set[Edge]] = {e: set() for e in boundary_edges}
    v_to_bedges: Dict[Vertex, List[Edge]] = defaultdict(list)
    for e in boundary_edges:
        u, v = e
        v_to_bedges[u].append(e)
        v_to_bedges[v].append(e)
    for edges in v_to_bedges.values():
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                a, b = edges[i], edges[j]
                adj[a].add(b)
                adj[b].add(a)

    comp_id = {}
    cid = 0
    for e in boundary_edges:
        if e in comp_id:
            continue
        # BFS on boundary-edge graph
        q = deque([e])
        comp_id[e] = cid
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in comp_id:
                    comp_id[y] = cid
                    q.append(y)
        cid += 1

    return boundary_edges, comp_id, cid

def spanning_forest(G: Dict[Vertex, Set[Vertex]]) -> Set[Edge]:
    """
    Build a primal spanning forest T ⊆ E(G).
    Returns the set of tree edges (undirected).
    """
    seen = set()
    T: Set[Edge] = set()
    for s in G:
        if s in seen:
            continue
        seen.add(s)
        q = deque([s])
        while q:
            u = q.popleft()
            for v in G[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
                    T.add(norm_edge(u, v))
    return T

def build_dual_graph(F: List[Face],
                     V: List[Vertex],
                     boundary_info=None):
    """
    Construct the dual graph G*.
    Nodes:
      - 0..F-1 for faces
      - F..F+B-1 : one 'exterior' node per boundary component (if any)
    Dual edges correspond 1-1 with primal edges.
    We return:
      - dual_adj: adjacency list of dual graph
      - dual_edge_map: map from primal edge -> (dual_u, dual_v)
      - B: number of boundary components used
    """
    if boundary_info is None:
        boundary_edges, edge2bc, B = set(), {}, 0
    else:
        boundary_edges, edge2bc, B = boundary_info

    inc = incident_faces_per_edge(F)
    n_faces = len(F)
    dual_adj: Dict[int, Set[int]] = {i: set() for i in range(n_faces + B)}
    dual_edge_map: Dict[Edge, Tuple[int, int]] = {}

    for e, faces in inc.items():
        if len(faces) == 2:
            f, g = faces
            dual_adj[f].add(g)
            dual_adj[g].add(f)
            dual_edge_map[e] = (f, g)
        elif len(faces) == 1:
            # boundary edge: connect face to its boundary component node
            f = faces[0]
            b = edge2bc[e]  # which boundary component
            ext_node = n_faces + b
            dual_adj[f].add(ext_node)
            dual_adj[ext_node].add(f)
            dual_edge_map[e] = (f, ext_node)
        else:
            # Non-manifold edge. You may want to raise here.
            raise ValueError(f"Non-manifold edge {e} incident to {len(faces)} faces.")
    return dual_adj, dual_edge_map, B

def dual_spanning_forest(dual_adj: Dict[int, Set[int]],
                         forbidden_dual_edges: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Build a spanning forest in the dual graph, avoiding any dual edge whose (u,v)
    (with u < v) is in forbidden_dual_edges.
    Returns the set of dual edges (u,v) with u < v used by the forest.
    """
    nodes = list(dual_adj.keys())
    seen = set()
    Tstar: Set[Tuple[int, int]] = set()

    for s in nodes:
        if s in seen:
            continue
        seen.add(s)
        q = deque([s])
        while q:
            u = q.popleft()
            for v in dual_adj[u]:
                a, b = (u, v) if u < v else (v, u)
                if v not in seen and (a, b) not in forbidden_dual_edges:
                    seen.add(v)
                    q.append(v)
                    Tstar.add((a, b))
    return Tstar

def build_tree_parent(G: Dict[Vertex, Set[Vertex]], T: Set[Edge]) -> Dict[Vertex, Vertex]:
    """
    Parent pointers for each tree in the forest T (arbitrary roots).
    Non-tree neighbors are ignored. Parents: parent[root] = root.
    """
    parent: Dict[Vertex, Vertex] = {}
    # Build adjacency restricted to T
    TG = {u: set() for u in G}
    for u, v in T:
        TG[u].add(v)
        TG[v].add(u)

    for s in TG:
        if s in parent:
            continue
        parent[s] = s
        q = deque([s])
        while q:
            u = q.popleft()
            for v in TG[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
    return parent

def path_in_tree(parent: Dict[Vertex, Vertex], a: Vertex, b: Vertex) -> List[Vertex]:
    """
    Return the unique vertex path from a to b inside the forest described by parent pointers.
    """
    # Build ancestors of a
    A = []
    x = a
    while True:
        A.append(x)
        if parent[x] == x:
            break
        x = parent[x]
    A_set = {x for x in A}
    # Walk up from b until we hit ancestor of a
    B_path = []
    y = b
    while y not in A_set:
        B_path.append(y)
        if parent[y] == y:
            break
        y = parent[y]
    lca = y
    # Path is: a -> ... -> lca, then reverse(B_path) to b
    path = []
    # a up to lca
    x = a
    while x != lca:
        path.append(x)
        x = parent[x]
    path.append(lca)
    # lca down to b
    for node in reversed(B_path):
        path.append(node)
    return path

def cycles_from_tree_cotree(V: List[Vertex], E: List[Edge], F: List[Face]):
    """
    Compute an H1 basis via tree–cotree.
    Returns:
      cycles_V: list of vertex-loops (each loop repeats the first vertex at the end),
      cycles_E: list of oriented edge loops (as (u,v) with sign via direction),
      rank: number of cycles
    """
    # Normalize inputs
    V = list(V)
    E = [norm_edge(u, v) for (u, v) in E]
    F = [tuple(f) for f in F]

    G = build_primal_graph(V, E)
    T = spanning_forest(G)  # primal spanning forest

    # Boundary handling (build exterior nodes per boundary component)
    b_edges, edge2bc, B = boundary_edges_and_components(V, F)
    dual_adj, dual_edge_map, _ = build_dual_graph(F, V, (b_edges, edge2bc, B))

    # Dual edges that are forbidden are those whose primal edge is already in T
    forbidden_dual_edges: Set[Tuple[int, int]] = set()
    for e, (u_star, v_star) in dual_edge_map.items():
        if e in T:
            a, b = (u_star, v_star) if u_star < v_star else (v_star, u_star)
            forbidden_dual_edges.add((a, b))

    Tstar = dual_spanning_forest(dual_adj, forbidden_dual_edges)

    # The leftover primal edges give H1 generators:
    # R = E \ (T ∪ {primal edges whose dual is in T*})
    dual_in_Tstar: Set[Tuple[int, int]] = Tstar
    R: List[Edge] = []
    for e in E:
        if e in T:
            continue
        u_star, v_star = dual_edge_map[e]
        a, b = (u_star, v_star) if u_star < v_star else (v_star, u_star)
        if (a, b) not in dual_in_Tstar:
            R.append(e)

    # Build parent pointers for T to recover unique tree paths
    parent = build_tree_parent(G, T)

    # Convert each e=(u,v) ∈ R into a vertex loop: path(u,v) in T plus the edge (v,u) to close
    cycles_V: List[List[Vertex]] = []
    cycles_E: List[List[Tuple[int, int]]] = []
    for (u, v) in R:
        path_uv = path_in_tree(parent, u, v)
        # Close with (v,u) to form a loop in vertex form
        loop_V = list(path_uv) + [u]  # path u..v then add u to close (since path ends at v)
        # Oriented edge sequence
        loop_E = []
        for a, b in zip(loop_V[:-1], loop_V[1:]):
            loop_E.append((a, b))  # orientation as we traverse
        cycles_V.append(loop_V)
        cycles_E.append(loop_E)

    rank = len(R)
    return cycles_V, cycles_E, rank