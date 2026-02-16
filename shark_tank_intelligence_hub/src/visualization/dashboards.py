import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class DashboardComponents:
    @staticmethod
    def create_metric_cards(col1, col2, col3, col4, metrics):
        with col1:
            st.metric(
                label="Total Pitches",
                value=metrics.get('total_pitches', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Success Rate",
                value=f"{metrics.get('success_rate', 0):.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Total Investment",
                value=f"₹{metrics.get('total_investment', 0):,.0f}L",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Avg Deal Size",
                value=f"₹{metrics.get('avg_deal_size', 0):.1f}L",
                delta=None
            )
    
    @staticmethod
    def create_filter_sidebar(df):
        st.sidebar.header("Filters")
        
        seasons = st.sidebar.multiselect(
            "Select Seasons",
            options=sorted(df['season'].unique()) if 'season' in df.columns else [],
            default=sorted(df['season'].unique()) if 'season' in df.columns else []
        )
        
        industries = st.sidebar.multiselect(
            "Select Industries",
            options=sorted(df['industry'].unique()) if 'industry' in df.columns else [],
            default=sorted(df['industry'].unique()) if 'industry' in df.columns else []
        )
        
        return {'seasons': seasons, 'industries': industries}
    
    @staticmethod
    def create_data_table(df, title="Data Overview"):
        st.subheader(title)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def create_download_button(df, filename="data.csv"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
    
    @staticmethod
    def create_comparison_chart(df, x_col, y_col, color_col=None, chart_type='bar'):
        if chart_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, color=color_col)
        elif chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, color=color_col)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        else:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col)
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_info_box(title, content, box_type='info'):
        if box_type == 'info':
            st.info(f"**{title}**\n\n{content}")
        elif box_type == 'success':
            st.success(f"**{title}**\n\n{content}")
        elif box_type == 'warning':
            st.warning(f"**{title}**\n\n{content}")
        elif box_type == 'error':
            st.error(f"**{title}**\n\n{content}")
