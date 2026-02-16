import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="Shark Networks", page_icon="🤝", layout="wide")

st.title("🤝 Shark Collaboration Network")
st.markdown("### Network Analysis of Shark Partnerships & Synergies")

st.markdown("""
Explore how sharks collaborate, their partnership patterns, and influence within the ecosystem.
""")

st.subheader("🦈 Shark Partnership Statistics")

partnerships_data = {
    'Partnership': [
        'Aman - Namita',
        'Peyush - Aman',
        'Vineeta - Namita',
        'Aman - Anupam',
        'Peyush - Namita'
    ],
    'Collaborations': [45, 38, 32, 28, 25],
    'Avg Investment': [85.5, 92.3, 78.4, 105.2, 88.7],
    'Success Rate': [78, 82, 75, 71, 80]
}

partnerships_df = pd.DataFrame(partnerships_data)

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(partnerships_df, use_container_width=True)

with col2:
    st.metric("Most Common Partnership", "Aman - Namita")
    st.metric("Total Collaborations", "168")
    st.metric("Avg Partnership Size", "33.6 deals")

st.markdown("---")

st.subheader("🕸️ Network Visualization")

G = nx.Graph()

sharks = ['Aman', 'Namita', 'Peyush', 'Vineeta', 'Anupam']
edges = [
    ('Aman', 'Namita', 45),
    ('Peyush', 'Aman', 38),
    ('Vineeta', 'Namita', 32),
    ('Aman', 'Anupam', 28),
    ('Peyush', 'Namita', 25),
    ('Vineeta', 'Aman', 22),
    ('Peyush', 'Anupam', 18),
    ('Vineeta', 'Peyush', 15)
]

for shark in sharks:
    G.add_node(shark)

for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

pos = nx.spring_layout(G, k=2, iterations=50)

edge_x = []
edge_y = []
edge_weights = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_weights.append(edge[2]['weight'])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=2, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_text = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}<br>Degree: {G.degree(node)}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=sharks,
    textposition='top center',
    hovertext=node_text,
    hoverinfo='text',
    marker=dict(
        size=40,
        color='lightblue',
        line=dict(width=2, color='darkblue')
    ))

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Shark Collaboration Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
                ))

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("📊 Centrality Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Degree Centrality**")
    st.markdown("(Most Connected)")
    st.write("1. Aman Gupta - 0.85")
    st.write("2. Namita Thapar - 0.78")
    st.write("3. Peyush Bansal - 0.72")

with col2:
    st.markdown("**Betweenness Centrality**")
    st.markdown("(Bridge Between Groups)")
    st.write("1. Aman Gupta - 0.42")
    st.write("2. Peyush Bansal - 0.35")
    st.write("3. Namita Thapar - 0.28")

with col3:
    st.markdown("**Influence Score**")
    st.markdown("(Overall Impact)")
    st.write("1. Aman Gupta - 0.88")
    st.write("2. Namita Thapar - 0.82")
    st.write("3. Peyush Bansal - 0.76")

st.markdown("---")

st.subheader("💡 Key Insights")

st.info("""
**Partnership Patterns:**
- Aman Gupta is the most collaborative shark, partnering in 45% of his deals
- Namita-Aman is the strongest partnership with 45 joint investments
- Tech deals see more Peyush-Aman collaborations
- Consumer goods attract Vineeta-Namita partnerships
""")

st.success("""
**Strategic Recommendations:**
- If targeting multiple sharks, pitch to Aman + Namita for highest success
- For tech startups, emphasize Peyush-Aman synergy
- For D2C brands, highlight Vineeta-Namita complementary expertise
""")
