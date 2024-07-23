from typing import Generator, Union

import networkx as nx


def get_paths(G: nx.DiGraph, s: Union[int, str], t: Union[int, str]) -> Generator:
    """Get the paths of graph G from source s to terminal t. """
    
    def DFS(start, path=[], visited=set()):
        path.append(start)
        if start in visited:
            return
        else:
            visited.add(start)
        if start == t:
            yield path
            return
        for neighbor in G.neighbors(start):
            yield from DFS(neighbor, path=path.copy(), visited=visited.copy())

    return DFS(s)

def DCF(t: float) -> float:
    """ Deprivation cost funcion based only on time.
    
    Values taken from Fig. 4, section 4: Numerical Experiments 
    of the paper of Cantillo.

    """
    return 0.0216255*(t**2) + 0.052425*t + 0.8272

def load_input_graph(file_path: str) -> nx.DiGraph:
    G = nx.DiGraph()
    with open(file_path, "r") as f:
        n_vertex, n_edges = map(int, f.readline().split())
        for line in f:
            line = line.split()
            u, v = line[:2]
            # weights
            time = float(line[2])
            logistic_cost = float(line[3]) if len(line) > 3 else None
            mode = line[4] if len(line) > 4 else None
            G.add_edge(u, v, time=time, logistic_cost=logistic_cost, mode=mode)
            

    if G.number_of_nodes() != n_vertex:
        raise ValueError(
            f"The graph contains {G.number_of_nodes()} and the file input says it has {n_vertex}"
        )
    if G.number_of_edges() != n_edges:
        raise ValueError(
            f"The graph contains {G.number_of_edges()} and the file input says it has {n_edges}"
        )
    
    return G
