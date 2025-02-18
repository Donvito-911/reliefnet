import sys
from typing import Generator, Optional, Union

import networkx as nx

from utils import get_paths, load_input_graph


def dial(
    G: nx.DiGraph,
    s: Union[int, str],
    t: Union[int, str],
    restriction: Optional[str] = "both",
) -> Generator:
    """execute Dial Algorithm to find reasonable paths

    :param G: Directed Graph
    :param s: source vertex
    :param t: terminal vertex
    :param restriction: two restrictions: 'further' for getting further away from source vertex, 'closer' for getting
    closer to terminal vertex. 'both' if both conditions must be True for reasonable paths and 'any' if either can be True.
    :returns: Acyclic graph with only the reasonable edges.

    """
    # drop the edges from terminal vertex, otherwise it could mess up the costs calculation via dijkstra
    G.remove_edges_from(list(G.out_edges(t)))

    # calculate reasonable edges from further condition
    if restriction in {"further", "both", "any"}:
        further_condition_edges = set()
        distance_from_origin, _ = nx.single_source_dijkstra(G, s, weight="time")
        for tail, head, time in G.edges(data="time"):
            if distance_from_origin[head] > distance_from_origin[tail]:
                further_condition_edges.add((tail, head, time))

    # calculate reasonable edges from closeness condition
    if restriction in {"closer", "both", "any"}:
        closeness_condition_edges = set()
        proximity_to_terminal, _ = nx.single_source_dijkstra(G.reverse(), t, weight="time")
        for tail, head, time in G.edges(data="time"):
            if proximity_to_terminal[head] < proximity_to_terminal[tail]:
                closeness_condition_edges.add((tail, head, time))

    if restriction == "both":
        reasonable_edges = set.intersection(
            further_condition_edges, closeness_condition_edges
        )
    elif restriction == "any":
        reasonable_edges = set.union(further_condition_edges, closeness_condition_edges)
    elif restriction == "further":
        reasonable_edges = further_condition_edges
    elif restriction == "closer":
        reasonable_edges = closeness_condition_edges
    else:
        raise Exception("Invalid restriction")

    # create the graph with reasonable edges (if restriction is not 'any', resulting graph is acyclic)
    G_ = nx.DiGraph()
    G_.add_weighted_edges_from(list(reasonable_edges))
    reasonable_paths = get_paths(G_, s, t)
    return reasonable_paths


def main(
    file_path: str,
    source_vertex: Union[int, str],
    terminal_vertex: Union[int, str],
    method: str,
):
    """execute Dial's algorithm for finding reasonable/efficient paths

    :param input_file:
    :param source_vertex:
    :param terminal_vertex:
    :param method: conditions for Dial's algorithm: 'further', 'closer' or 'both'. Default is 'both'
    :returns:

    """
    G = load_input_graph(file_path)
    if source_vertex not in G.nodes():
        raise ValueError("The graph does not contain the source vertex")
    if terminal_vertex not in G.nodes():
        raise ValueError("The graph does not contain the terminal vertex")
    for i, reasonable_path in enumerate(
        dial(G, source_vertex, terminal_vertex, method), 1
    ):
        print(f"Path {i}: {reasonable_path}")


if __name__ == "__main__":
    if len(sys.argv) not in range(4, 6):
        raise Exception(
            "Incorrect use of script. python dial2.py <input_file> <source_vertex> <terminal_vertex> <optional: method>"
        )

    input_file = sys.argv[1]
    source_vertex = sys.argv[2]
    terminal_vertex = sys.argv[3]
    method = sys.argv[4] if len(sys.argv) == 5 else "both"
    main(input_file, source_vertex, terminal_vertex, method)
