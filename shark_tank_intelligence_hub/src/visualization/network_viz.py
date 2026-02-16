import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

class NetworkVisualizer:
    def __init__(self):
        self.layout_algorithms = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }
        
    def plot_shark_network_plotly(self, graph, layout='spring'):
        if len(graph.nodes()) == 0:
            return None
        
        layout_func = self.layout_algorithms.get(layout, nx.spring_layout)
        pos = layout_func(graph)
        
        edge_trace = []
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 1)
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=weight/2, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Degree: {graph.degree(node)}")
            node_size.append(20 + graph.degree(node) * 5)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[node for node in graph.nodes()],
            textposition='top center',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color='#1f77b4',
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title='Shark Collaboration Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def plot_shark_network_matplotlib(self, graph, figsize=(12, 8)):
        if len(graph.nodes()) == 0:
            return None
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        edges = graph.edges()
        weights = [graph[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_nodes(
            graph, pos,
            node_color='lightblue',
            node_size=3000,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            graph, pos,
            width=[w/2 for w in weights],
            alpha=0.5,
            edge_color='gray'
        )
        
        nx.draw_networkx_labels(
            graph, pos,
            font_size=10,
            font_weight='bold'
        )
        
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        plt.title('Shark Collaboration Network', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_centrality_visualization(self, graph, centrality_metric):
        if len(graph.nodes()) == 0:
            return None
        
        pos = nx.spring_layout(graph)
        
        node_colors = [centrality_metric.get(node, 0) for node in graph.nodes()]
        
        plt.figure(figsize=(12, 8))
        
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=3000,
            cmap=plt.cm.viridis,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(graph, pos, alpha=0.3)
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Centrality Score')
        plt.title('Shark Network Centrality', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
