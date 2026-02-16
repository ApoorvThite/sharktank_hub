import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations

class NetworkAnalyzer:
    def __init__(self):
        self.graph = nx.Graph()
        self.shark_stats = {}
        
    def build_shark_network(self, df, shark_columns):
        for _, row in df.iterrows():
            active_sharks = [shark for shark in shark_columns if row.get(shark, 0) == 1]
            
            if len(active_sharks) > 1:
                for shark1, shark2 in combinations(active_sharks, 2):
                    if self.graph.has_edge(shark1, shark2):
                        self.graph[shark1][shark2]['weight'] += 1
                    else:
                        self.graph.add_edge(shark1, shark2, weight=1)
            
            for shark in active_sharks:
                if shark not in self.graph:
                    self.graph.add_node(shark)
        
        return self.graph
    
    def calculate_centrality_metrics(self):
        metrics = {}
        
        if len(self.graph.nodes()) > 0:
            metrics['degree_centrality'] = nx.degree_centrality(self.graph)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
            
            if nx.is_connected(self.graph):
                metrics['closeness_centrality'] = nx.closeness_centrality(self.graph)
            else:
                metrics['closeness_centrality'] = {}
                for component in nx.connected_components(self.graph):
                    subgraph = self.graph.subgraph(component)
                    metrics['closeness_centrality'].update(nx.closeness_centrality(subgraph))
        
        return metrics
    
    def get_shark_partnerships(self):
        partnerships = []
        
        for edge in self.graph.edges(data=True):
            partnerships.append({
                'shark1': edge[0],
                'shark2': edge[1],
                'collaborations': edge[2]['weight']
            })
        
        return sorted(partnerships, key=lambda x: x['collaborations'], reverse=True)
    
    def get_network_statistics(self):
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if len(self.graph.nodes()) > 1 else 0,
            'num_components': nx.number_connected_components(self.graph)
        }
        
        if len(self.graph.nodes()) > 0:
            stats['avg_degree'] = sum(dict(self.graph.degree()).values()) / len(self.graph.nodes())
        else:
            stats['avg_degree'] = 0
        
        return stats
    
    def find_communities(self):
        if len(self.graph.nodes()) > 2:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(self.graph)
            return [list(c) for c in communities]
        return []
    
    def get_shark_influence_score(self):
        centrality_metrics = self.calculate_centrality_metrics()
        
        influence_scores = {}
        for shark in self.graph.nodes():
            score = 0
            if 'degree_centrality' in centrality_metrics:
                score += centrality_metrics['degree_centrality'].get(shark, 0) * 0.4
            if 'betweenness_centrality' in centrality_metrics:
                score += centrality_metrics['betweenness_centrality'].get(shark, 0) * 0.3
            if 'closeness_centrality' in centrality_metrics:
                score += centrality_metrics['closeness_centrality'].get(shark, 0) * 0.3
            
            influence_scores[shark] = score
        
        return sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
