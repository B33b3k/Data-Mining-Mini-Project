import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import community as community_louvain
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import logging
from typing import Dict, Any, Tuple

# --- ======================================================= ---
# ---              LOGGING & APP CONFIGURATION                ---
# --- ======================================================= ---


# --- ======================================================= ---
# ---                MODEL TRAINING FUNCTIONS                ---
# --- ======================================================= ---

@st.cache_data
def train_cora_models(content_df: pd.DataFrame) -> Dict[str, Any]:
    """Trains and evaluates multiple models for Cora subject classification."""
    logger.info("Starting Cora model training...")
    df = content_df.dropna(subset=['text', 'subject'])
    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    y = df['subject']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "SGD Classifier": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42, class_weight='balanced')
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name} for Cora...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=model.classes_)
        results[name] = {'model': model, 'accuracy': accuracy, 'confusion_matrix': cm, 'vectorizer': vectorizer}
    logger.info("Cora model training finished.")
    return results

@st.cache_data
def train_github_models(master_df: pd.DataFrame) -> Dict[str, Any]:
    """Trains and evaluates multiple models for GitHub developer classification."""
    logger.info("Starting GitHub model training...")
    features = master_df[['Degree', 'Betweenness', 'PageRank', 'Community']]
    labels = master_df['is_ml_dev']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        logger.info(f"Training {name} for GitHub...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        importances = model.feature_importances_
        results[name] = {'model': model, 'accuracy': accuracy, 'confusion_matrix': cm, 'importances': importances}
    logger.info("GitHub model training finished.")
    return results

# --- ======================================================= ---
# ---              UI & VISUALIZATION FUNCTIONS               ---
# --- ======================================================= ---

def draw_network_graph(G: nx.Graph, df: pd.DataFrame, config: dict, num_nodes: int):
    """Generates an interactive Plotly graph of the network."""
    with st.spinner(f"Generating network graph for top {num_nodes} nodes..."):
        top_nodes = df.nlargest(num_nodes, "Degree")[config['id_col']].tolist()
        subgraph = G.subgraph(top_nodes)
        
        pos = nx.spring_layout(subgraph, seed=42)
        
        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        subgraph_df = df[df[config['id_col']].isin(top_nodes)].set_index(config['id_col'])
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_info = subgraph_df.loc[node]
            text = f"{config['id_col']}: {node}<br>{config['name_col']}: {node_info[config['name_col']]}<br>Degree: {node_info['Degree']}<br>Community: {node_info['Community']}"
            node_text.append(text)
            node_color.append(node_info['Community'])
            node_size.append(node_info['Degree'])

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
            marker=dict(
                showscale=True, colorscale='viridis', reversescale=True, color=node_color,
                size=node_size, sizemin=4, sizemode='area', line_width=2
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'Network of Top {num_nodes} Nodes by Degree', showlegend=False, hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        st.plotly_chart(fig, use_container_width=True)

def render_overview_page(df: pd.DataFrame, G: nx.Graph, config: dict):
    st.subheader("📊 Exploratory Data Analysis")
    st.markdown("A high-level summary of the network's structure and composition.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", f"{G.number_of_nodes():,}")
    col2.metric("Total Edges", f"{G.number_of_edges():,}")
    col3.metric("Detected Communities", f"{df['Community'].nunique():,}")
    col4.metric("Graph Density", f"{nx.density(G):.6f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Node Type Distribution**")
        fig = px.pie(df, names=config['name_col'], title=f"Composition of {config['name_col']}s")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"**Network Degree Distribution**")
        fig = px.histogram(df, x="Degree", title="Degree Distribution (Log Scale)", log_y=True)
        st.plotly_chart(fig, use_container_width=True)

def render_sna_page(df: pd.DataFrame, G: nx.Graph, config: dict):
    st.subheader("🔬 Centrality and Connectivity")
    st.markdown("Analyze node importance and visualize the network's core structure.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Most Influential Nodes**")
        metric = st.selectbox("Rank nodes by:", config['centrality_metrics'], help="Select a metric to rank nodes.", key=f"metric_{config['id_col']}")
        st.dataframe(df.sort_values(by=metric, ascending=False).head(10)[
            [config['id_col'], config['name_col'], 'Community'] + config['centrality_metrics']
        ])
    with c2:
        st.markdown("**Centrality Correlation**")
        x_axis = st.selectbox("X-axis", config['centrality_metrics'], index=0, key=f"x_{config['id_col']}")
        y_axis = st.selectbox("Y-axis", config['centrality_metrics'], index=1, key=f"y_{config['id_col']}")
        fig = px.scatter(df.sample(min(1000, len(df))), x=x_axis, y=y_axis, hover_name=config['id_col'],
                         title=f"Correlation: {x_axis} vs. {y_axis}", opacity=0.5,
                         color='Community', color_continuous_scale=px.colors.cyclical.IceFire)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Interactive Network Graph**")
    num_nodes = st.slider("Number of nodes to display:", 50, 500, 200, 50, key=f"slider_{config['id_col']}")
    draw_network_graph(G, df, config, num_nodes)

def render_communities_page(df: pd.DataFrame, config: dict):
    st.subheader("👨‍👩‍👧‍👦 Community Deep-Dive")
    st.markdown("Explore the distinct communities detected within the network.")
    
    communities_by_size = df['Community'].value_counts()
    selected_community = st.selectbox("Select a Community to Analyze:", options=communities_by_size.index, key=f"comm_{config['id_col']}")
    
    community_df = df[df['Community'] == selected_community]
    
    st.markdown(f"**Analysis of Community {selected_community}** ({len(community_df)} members)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Share of Total Network", f"{len(community_df)/len(df):.2%}")
        st.metric("Avg. Degree", f"{community_df['Degree'].mean():.2f}")
    with c2:
        fig = px.pie(community_df, names=config['name_col'], title=f"Composition of Community {selected_community}")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Community Leaders (Most Central Members)**")
    st.dataframe(community_df.sort_values(by='Degree', ascending=False).head(10)[
        [config['id_col'], config['name_col']] + config['centrality_metrics']
    ])

def render_kcore_page(G: nx.Graph, config: dict):
    st.subheader("👑 K-Core Decomposition")
    st.markdown("A k-core is the largest subgraph where every node has at least `k` connections within the subgraph. It helps find the most resilient and dense parts of a network.")
    
    max_k = max(nx.core_number(G).values()) if G.nodes else 1
    k_value = st.slider("Select a value for k:", 1, max_k, max(1, max_k // 2), key=f"kcore_{config['id_col']}")
    
    cores = nx.k_core(G, k=k_value)
    df_cores = pd.DataFrame(nx.core_number(G).items(), columns=['Node', 'CoreNumber'])
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"Nodes in {k_value}-core", f"{cores.number_of_nodes():,}")
        st.metric(f"Edges in {k_value}-core", f"{cores.number_of_edges():,}")
    with c2:
        st.markdown("**Core Number Distribution**")
        fig = px.histogram(df_cores, x='CoreNumber', title="Distribution of Core Numbers")
        st.plotly_chart(fig, use_container_width=True)

def render_modeling_page(results: dict, config: dict):
    st.subheader("🤖 Predictive Model Comparison")
    st.markdown("Train and compare models to predict node attributes based on their features.")
    
    st.markdown("**Performance Summary**")
    perf_data = {name: res['accuracy'] for name, res in results.items()}
    perf_df = pd.DataFrame.from_dict(perf_data, orient='index', columns=['Accuracy'])
    st.table(perf_df.style.format({'Accuracy': '{:.2%}'}).highlight_max(axis=0, color='lightgreen'))

    selected_model = st.selectbox("Choose model to inspect:", list(results.keys()), key=f"model_{config['id_col']}")
    res = results[selected_model]
    
    st.markdown(f"**Inspection: {selected_model}**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix**")
        labels = list(res['model'].classes_)
        fig = ff.create_annotated_heatmap(res['confusion_matrix'], x=labels, y=labels, colorscale='Blues', showscale=True)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        if 'importances' in res: # For Tree-based models (GitHub)
            st.markdown("**Feature Importances**")
            imp_df = pd.DataFrame({'Feature': config['model_features'], 'Importance': res['importances']})
            fig = px.bar(imp_df.sort_values('Importance', ascending=False), x='Feature', y='Importance')
            st.plotly_chart(fig, use_container_width=True)
        elif 'vectorizer' in res: # For text models (Cora)
            st.markdown("**Top Words per Subject**")
            vectorizer = res['vectorizer']
            model = res['model']
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            with st.expander("See Top 15 influential words for each category", expanded=True):
                for i, class_label in enumerate(model.classes_):
                    top_indices = np.argsort(model.coef_[i])[-15:]
                    top_features = feature_names[top_indices]
                    st.markdown(f"**{class_label}**: `{'`, `'.join(reversed(top_features))}`")

# --- ======================================================= ---
# ---                MAIN APPLICATION LOGIC                   ---
# --- ======================================================= ---

APP_CONFIG = {
    "Cora": {
        "title": "📖 Research Paper Network (Cora)",
        "loader": load_cora_data, "analyzer": analyze_cora_network, "modeler": train_cora_models,
        "id_col": "paper_id", "name_col": "subject", "centrality_metrics": ['Degree', 'In-Degree', 'PageRank'],
        "files": {"content_file": Path("cora/cora.content"), "cites_file": Path("cora/cora.cites")},
        "model_df_source": "content"
    },
    "GitHub": {
        "title": "💻 Developer Social Network (GitHub)",
        "loader": load_github_data, "analyzer": analyze_github_network, "modeler": train_github_models,
        "id_col": "user_id", "name_col": "dev_type", "centrality_metrics": ['Degree', 'Betweenness', 'PageRank'],
        "model_features": ['Degree', 'Betweenness', 'PageRank', 'Community'],
        "files": {"edges_file": Path("git_web_ml/musae_git_edges.csv"), "target_file": Path("git_web_ml/musae_git_target.csv")},
        "model_df_source": "master"
    }
}

def run_dataset_pipeline(config: Dict[str, Any]) -> Tuple[pd.DataFrame, nx.Graph, Dict[str, Any]]:
    """Loads, analyzes, and models a single dataset based on its config."""
    data_files = list(config['files'].values())
    df1, df2 = config['loader'](*data_files)
    G, master_df = config['analyzer'](df1, df2)
    model_df = df1 if config['model_df_source'] == "content" else master_df
    model_results = config['modeler'](model_df)
    return master_df, G, model_results

def main():
    """Main function to run the Streamlit app with a tabbed interface."""
    logger.info("Application starting.")
    st.sidebar.title("Network Analysis Dashboard")
    st.sidebar.markdown("An interactive tool to explore and compare complex networks.")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Analysis View:",
        ["Overview", "Centrality and Connectivity", "Community Deep-Dive", "K-Core Decomposition", "Model Comparison"]
    )
    st.sidebar.markdown("---")
    st.sidebar.info("The app is now tab-based. Select a view above and compare the datasets side-by-side.")

    tab1, tab2 = st.tabs([APP_CONFIG['Cora']['title'], APP_CONFIG['GitHub']['title']])

    # --- Cora Tab ---
    with tab1:
        cora_config = APP_CONFIG['Cora']
        if not all(p.exists() for p in cora_config['files'].values()):
            st.error("Cora data files not found. Please check your `cora/` directory.")
            logger.warning("Missing Cora data files.")
        else:
            try:
                master_df, G, model_results = run_dataset_pipeline(cora_config)
                if page == "Overview":
                    render_overview_page(master_df, G, cora_config)
                elif page == "Centrality and Connectivity":
                    render_sna_page(master_df, G, cora_config)
                elif page == "Community Deep-Dive":
                    render_communities_page(master_df, cora_config)
                elif page == "K-Core Decomposition":
                    render_kcore_page(G.to_undirected(), cora_config)
                elif page == "Model Comparison":
                    render_modeling_page(model_results, cora_config)
            except Exception as e:
                st.error(f"An error occurred while processing the Cora dataset: {e}")
                logger.error("Cora pipeline failed", exc_info=True)

    # --- GitHub Tab ---
    with tab2:
        github_config = APP_CONFIG['GitHub']
        if not all(p.exists() for p in github_config['files'].values()):
            st.error("GitHub data files not found. Please check that `musae_git_edges.csv` and `musae_git_target.csv` exist in the `git_web_ml/` directory.")
            logger.warning("Missing GitHub data files.")
        else:
            try:
                master_df, G, model_results = run_dataset_pipeline(github_config)
                if page == "Overview":
                    render_overview_page(master_df, G, github_config)
                elif page == "Centrality and Connectivity":
                    render_sna_page(master_df, G, github_config)
                elif page == "Community Deep-Dive":
                    render_communities_page(master_df, github_config)
                elif page == "K-Core Decomposition":
                    render_kcore_page(G, github_config)
                elif page == "Model Comparison":
                    render_modeling_page(model_results, github_config)
            except Exception as e:
                st.error(f"An error occurred while processing the GitHub dataset: {e}")
                logger.error("GitHub pipeline failed", exc_info=True)

if __name__ == "__main__":
    main()