"""NetworkX-based co-investment graph analysis for Shark Tank India sharks."""

import logging
from itertools import combinations
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """Builds and analyses the shark co-investment network.

    The network is an undirected weighted graph where:
    - **Nodes** represent individual sharks.
    - **Edges** represent co-investment relationships.
    - **Edge weights** count the number of deals two sharks did together.

    Attributes
    ----------
    graph : nx.Graph
        The co-investment network graph.
    shark_stats : Dict
        Placeholder for per-shark aggregate statistics.
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.shark_stats: Dict[str, Any] = {}

    def build_shark_network(
        self, df: pd.DataFrame, shark_columns: List[str]
    ) -> nx.Graph:
        """Construct the co-investment graph from the deals dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset where each row is a deal and shark columns contain 0/1
            flags indicating participation.
        shark_columns : List[str]
            Column names corresponding to each shark.

        Returns
        -------
        nx.Graph
            Populated co-investment graph.
        """
        for _, row in df.iterrows():
            active_sharks = [s for s in shark_columns if row.get(s, 0) == 1]

            for shark in active_sharks:
                if shark not in self.graph:
                    self.graph.add_node(shark)

            for shark1, shark2 in combinations(active_sharks, 2):
                if self.graph.has_edge(shark1, shark2):
                    self.graph[shark1][shark2]['weight'] += 1
                else:
                    self.graph.add_edge(shark1, shark2, weight=1)

        logger.info(
            "Network built: %d nodes, %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute degree, betweenness, and closeness centrality for all nodes.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Keys: ``degree_centrality``, ``betweenness_centrality``,
            ``closeness_centrality``. Values are node→score mappings.
        """
        metrics: Dict[str, Dict[str, float]] = {}
        if not self.graph.nodes():
            return metrics

        metrics['degree_centrality'] = nx.degree_centrality(self.graph)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(self.graph)

        if nx.is_connected(self.graph):
            metrics['closeness_centrality'] = nx.closeness_centrality(self.graph)
        else:
            closeness: Dict[str, float] = {}
            for component in nx.connected_components(self.graph):
                subgraph = self.graph.subgraph(component)
                closeness.update(nx.closeness_centrality(subgraph))
            metrics['closeness_centrality'] = closeness

        return metrics

    def get_shark_partnerships(self) -> List[Dict[str, Any]]:
        """Return all shark pairs sorted by number of co-investments descending.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict has keys ``shark1``, ``shark2``, and ``collaborations``.
        """
        partnerships = [
            {
                'shark1': u,
                'shark2': v,
                'collaborations': data['weight'],
            }
            for u, v, data in self.graph.edges(data=True)
        ]
        return sorted(partnerships, key=lambda x: x['collaborations'], reverse=True)

    def get_network_statistics(self) -> Dict[str, Any]:
        """Return high-level network statistics.

        Returns
        -------
        Dict[str, Any]
            Keys: ``num_nodes``, ``num_edges``, ``density``,
            ``num_components``, ``avg_degree``.
        """
        n_nodes = self.graph.number_of_nodes()
        stats: Dict[str, Any] = {
            'num_nodes': n_nodes,
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if n_nodes > 1 else 0.0,
            'num_components': nx.number_connected_components(self.graph),
            'avg_degree': (
                sum(dict(self.graph.degree()).values()) / n_nodes if n_nodes > 0 else 0.0
            ),
        }
        return stats

    def find_communities(self) -> List[List[str]]:
        """Detect communities using greedy modularity maximisation.

        Returns
        -------
        List[List[str]]
            List of communities, each being a list of shark names.
            Returns an empty list if the graph has fewer than 3 nodes.
        """
        if self.graph.number_of_nodes() <= 2:
            return []
        from networkx.algorithms import community  # noqa: PLC0415
        communities = community.greedy_modularity_communities(self.graph)
        return [list(c) for c in communities]

    def get_shark_influence_score(self) -> List[Tuple[str, float]]:
        """Compute a composite influence score for each shark.

        The score is a weighted combination of:
        - Degree centrality (40%)
        - Betweenness centrality (30%)
        - Closeness centrality (30%)

        Returns
        -------
        List[Tuple[str, float]]
            ``(shark_name, score)`` pairs sorted descending by score.
        """
        centrality = self.calculate_centrality_metrics()
        influence_scores: Dict[str, float] = {}

        for shark in self.graph.nodes():
            score = (
                centrality.get('degree_centrality', {}).get(shark, 0.0) * 0.4
                + centrality.get('betweenness_centrality', {}).get(shark, 0.0) * 0.3
                + centrality.get('closeness_centrality', {}).get(shark, 0.0) * 0.3
            )
            influence_scores[shark] = score

        return sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
