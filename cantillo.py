import itertools
import sys
from typing import Any, Iterable, List, Union, Tuple, Optional

import numpy as np
import networkx as nx

from dial import dial
from utils import DCF, load_input_graph


def cantillo(
    G: nx.DiGraph,
    sources: List[Any],
    terminals: List[Any],
    dial_method: str,
    disruptions: List[Tuple],
):
    # calculate expected social costs without disruptions
    expected_social_costs_before = []
    for s, t in itertools.product(sources, terminals):
        reasonable_paths = dial(G, s, t, dial_method)
        social_costs = get_social_costs(G, reasonable_paths)
        # Expected Social Cost
        ESC_0 = logsum_term(social_costs)
        expected_social_costs_before.append(ESC_0)
    expected_social_costs_before = np.array(expected_social_costs_before)

    # calculate expected social costs for each disruption
    for disruption in disruptions:
        G_disrupted = G.copy()
        G_disrupted.remove_edge(*disruption)
        expected_social_costs_after = []
        for s, t in itertools.product(sources, terminals):
            reasonable_paths = dial(G_disrupted, s, t, dial_method)
            social_costs = get_social_costs(G_disrupted, reasonable_paths)
            # Expected Social Cost
            ESC_1 = logsum_term(social_costs)
            expected_social_costs_after.append(ESC_1)
        expected_social_costs_after = np.array(expected_social_costs_after)
        delta_expected_social_costs = (
            expected_social_costs_after - expected_social_costs_before
        )
        # vulnerability indicator over all sources and terminals
        I_disruption = np.sum(
            delta_expected_social_costs / expected_social_costs_before
        )
        print(f"Disruption {disruption} I_s={I_disruption}")


def get_social_costs(G: nx.Graph, paths: Iterable) -> np.array:
    """calculate social cost for each reasonable path"""
    social_costs = []
    for path in paths:
        travel_time = 0
        logistic_cost = 0
        edges = zip(path, path[1:])  # convert the path to a list of edges
        # get the travel time and logistic costs of the path
        for u, v in edges:
            travel_time += G[u][v]["time"]
            logistic_cost += G[u][v]["logistic_cost"]
            
        social_cost = get_social_cost(travel_time, logistic_cost)
        social_costs.append(social_cost)
    social_costs = np.array(social_costs)
    return social_costs

def get_social_cost(travel_time: float, logistic_cost: float):
    """ calculate social cost given the travel time and the logistic cost of a path.

    Taken from eq. (1) in section 3: The proposed vulnerability model from the paper Cantillo. 
    """

    # El número 5 se infiere del paper que son 5 litros por persona de la DCF.
    return 5*DCF(travel_time) + logistic_cost

def logsum_term(
    social_costs: np.array,
    dispersion_param: Optional[float] = 0.426
) -> float:
    """calculate the log-sum term
    
    Eq.7 of Section 3:The Proposed Model. Dispersion param was used a proxy of Ë because, 
    according to the paper: As μ is nonnegative, the probability of using a particular 
    path is directly proportional to exp. (-μVijpgq)

    """
    
    return (1 / dispersion_param) * np.log(
        np.sum(np.exp(-dispersion_param * social_costs))
    )


def main(
    file_path: str,
    source_vertex: Union[int, str],
    terminal_vertex: Union[int, str],
    method: str,
    disruptions,
):
    G = load_input_graph(file_path)
    cantillo(G, source_vertex, terminal_vertex, method, disruptions)


if __name__ == "__main__":
    # if len(sys.argv) not in range(4, 6):
    #     raise Exception(
    #         "Incorrect use of script. python dial2.py <input_file> <source_vertex> <terminal_vertex> <optional: method>"
    #     )

    # input_file = sys.argv[1]
    input_file = "tests/cantillo_test1.in"
    # source_vertex = sys.argv[2].split(",")
    sources = ["1"]
    # terminal_vertex = sys.argv[3].split(",")
    terminals = ["9"]
    disruptions = [
        ("5", "9"),
        ("1", "2"),
        ("1", "4"),
        ("1", "5"),
        ("2", "5"),
        ("4", "5"),
        ("5", "2"),
        ("5", "4"),
        ("5", "6"),
        ("6", "9"),
        ("5", "8"),
        ("8", "9"),
        ("9", "6"),
        ("9", "5"),
    ]
    # method = sys.argv[4] if len(sys.argv) == 5 else "both"
    method = "any"
    main(input_file, sources, terminals, method, disruptions)
