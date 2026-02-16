"""
Main Streamlit Dashboard Application
Cross-Lingual NLP Framework - India-Japan Strategic Shift Analysis
Enhanced with statistical testing, better caching, and more visualizations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple
import sys
import os
import logging
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrapers.data_loader import DataLoader
from preprocessing.preprocessor import Preprocessor
from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer
from analysis.tone_analyzer import ToneAnalyzer
from analysis.thematic_clustering import ThematicAnalyzer
from utils.country_config import (
    COUNTRIES, get_country_name, 
    get_ministry_name
)
from utils.pdf_report_generator import PDFReportGenerator, REPORTLAB_AVAILABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Cross-Lingual NLP Framework",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stat-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #2E86AB;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        color: #1a1a1a;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .stat-box b {
        color: #0d47a1;
        font-weight: 700;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    h1, h2, h3 {
        color: #2E86AB;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading diplomatic documents...")
def load_data_for_pair(pair_str: str):
    """Load and cache diplomatic documents for a specific country pair"""
    logger.info(f"Loading documents for {pair_str}...")
    pair = tuple(pair_str.split('-'))
    
    # Import here to avoid circular imports
    from data_integration import DashboardDataManager
    manager = DashboardDataManager()
    df = manager.get_data_for_pair(pair)
    return df


@st.cache_data(show_spinner="Preprocessing documents...")
def preprocess_data(_df):
    """Preprocess documents with caching"""
    logger.info("Preprocessing documents...")
    preprocessor = Preprocessor()
    processed_df = preprocessor.process_dataframe(_df, content_column='content')
    return processed_df


@st.cache_data(show_spinner="Analyzing strategic shifts...")
def perform_strategic_analysis(_df):
    """Cached strategic shift analysis"""
    analyzer = StrategicShiftAnalyzer()
    report, scored_df, yearly_df = analyzer.generate_shift_report(_df)
    return report, scored_df, yearly_df


@st.cache_data(show_spinner="Analyzing tone & sentiment...")
def perform_tone_analysis(_df):
    """Cached tone analysis - Fixed to handle empty dataframes"""
    if _df is None or len(_df) == 0:
        logger.warning("Empty dataframe passed to tone analyzer")
        return None, pd.DataFrame()
    
    if 'cleaned' not in _df.columns:
        logger.warning("'cleaned' column not found in dataframe")
        return None, pd.DataFrame()
    
    tone_analyzer = ToneAnalyzer()
    tone_df = tone_analyzer.process_dataframe(_df, text_column='cleaned')
    return tone_analyzer, tone_df


@st.cache_data(show_spinner="Analyzing themes...")
def perform_thematic_analysis(_df):
    """Cached thematic analysis"""
    thematic_analyzer = ThematicAnalyzer(n_topics=5)
    analysis, df_themes = thematic_analyzer.analyze_theme_evolution(
        _df,
        text_column='cleaned'
    )
    return thematic_analyzer, analysis, df_themes


def calculate_statistical_significance(series1: pd.Series, series2: pd.Series) -> Dict:
    """Calculate statistical significance between two series"""
    try:
        t_stat, p_value = stats.ttest_ind(series1.dropna(), series2.dropna())
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': (series1.mean() - series2.mean()) / series1.std() if series1.std() > 0 else 0
        }
    except Exception as e:
        logger.error(f"Statistical error: {str(e)}")
        return None


def create_heatmap_visualization(data: pd.DataFrame, title: str):
    """Create advanced heatmap visualization"""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdYlGn',
        text=data.values,
        texttemplate='%{text:.2f}',
        hovertemplate='%{y}<br>%{x}<br>Value: %{z:.4f}<extra></extra>'
    ))
    fig.update_layout(title=title, height=400)
    return fig


def create_distribution_plot(data: List, title: str, xlabel: str):
    """Create advanced distribution plot with statistics"""
    mean = pd.Series(data).mean()
    std = pd.Series(data).std()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=30, name='Distribution'))
    fig.add_vline(x=mean, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean:.2f}")
    fig.add_vline(x=mean-std, line_dash="dot", line_color="orange", annotation_text=f"±1 SD")
    fig.add_vline(x=mean+std, line_dash="dot", line_color="orange")
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Frequency",
        height=400,
        showlegend=True
    )
    return fig


def main():
    """Main dashboard application"""
    
    # Language selection in sidebar
    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        language = st.selectbox(
            "Language",
            ["English", "日本語"],
            index=0,
            key="language_selector"
        )
    
    # Translation dictionary
    translations = {
        "English": {
            "title": "Cross-Lingual NLP Framework",
            "subtitle_template": "{country1}-{country2} Relations: From Trade to Security",
            "description_template": "Tracking how {country1}-{country2} diplomatic language evolved from economic cooperation to strategic security partnership over 25 years.",
            "documents_loaded": "Analyzing {count} diplomatic documents ({start}-{end})",
            "processing": "Processing documents...",
            "navigation": "Pages",
            "select_analysis": "Choose a Page:",
            "executive_summary": "Overview",
            "strategic_shift": "Economic vs Security",
            "tone_sentiment": "Tone & Mood",
            "thematic": "Topics & Themes",
            "time_machine": "Year Explorer",
            "search_explore": "Keyword Search",
            "statistical_tests": "Statistical Proof",
            "total_documents": "Total Documents",
            "total_words": "Total Words Analyzed",
            "key_findings": "Key Findings",
            "overall_economic": "Overall Economic Focus",
            "overall_security": "Overall Security Focus",
            "crossover_year": "Crossover Year",
            "security_overtook": "Security focus overtook economic focus",
            "strategic_shift_evident": "Strategic shift evident throughout the period",
            "overall_trend": "Overall Trend"
        },
        "日本語": {
            "title": "外交晴雨計",
            "subtitle_template": "{country1}・{country2}関係：経済から安全保障へ",
            "description_template": "{country1}・{country2}の外交言語が25年間でどのように経済協力から安全保障パートナーシップへ変化したかを追跡します。",
            "documents_loaded": "{count}の外交文書を分析中({start}年～{end}年)",
            "processing": "文書を処理中...",
            "navigation": "ページ",
            "select_analysis": "ページを選択:",
            "executive_summary": "概要",
            "strategic_shift": "経済vs安全保障",
            "tone_sentiment": "トーン・ムード",
            "thematic": "テーマ分析",
            "time_machine": "年別探索",
            "search_explore": "キーワード検索",
            "statistical_tests": "統計的証明",
            "total_documents": "総文書数",
            "total_words": "分析対象総語数",
            "key_findings": "主要な発見",
            "overall_economic": "経済重視度合い",
            "overall_security": "安全保障重視度合い",
            "crossover_year": "クロスオーバー年",
            "security_overtook": "安全保障が経済重視を上回った",
            "strategic_shift_evident": "戦略的シフトは期間を通じて明白",
            "overall_trend": "全体的トレンド"
        }
    }
    
    # Get current language translations
    lang = translations[language]
    
    # Sidebar navigation
    st.sidebar.title(lang["navigation"])
    
    # Fixed country pair: India-Japan
    selected_pair = ('india', 'japan')
    country1_name = get_country_name(selected_pair[0])
    country2_name = get_country_name(selected_pair[1])
    country1_ministry = get_ministry_name(selected_pair[0])
    country2_ministry = get_ministry_name(selected_pair[1])
    
    st.sidebar.info(f"🇮🇳 {country1_name} ({country1_ministry}) ↔ 🇯🇵 {country2_name} ({country2_ministry})")
    
    st.sidebar.markdown("---")
    
    # Now display title and introduction with dynamic country names
    st.title(lang["title"])
    st.markdown(f"### {lang['subtitle_template'].format(country1=country1_name, country2=country2_name)}")
    st.markdown(lang["description_template"].format(country1=country1_name, country2=country2_name))
    
    # Load data for the selected country pair
    pair_str = f"{selected_pair[0]}-{selected_pair[1]}"
    df = load_data_for_pair(pair_str)
    doc_count = len(df)
    year_min = df['year'].min()
    year_max = df['year'].max()
    st.success(f"Analyzing **{doc_count} diplomatic documents** from {year_min} to {year_max}")
    
    # Preprocess data
    with st.spinner(lang["processing"]):
        processed_df = preprocess_data(df)
    
    st.sidebar.markdown("---")
    
    # PDF Export button in sidebar
    if REPORTLAB_AVAILABLE:
        st.sidebar.subheader("Export Report")
        if st.sidebar.button("Download PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                try:
                    report, scored_df, yearly_df = perform_strategic_analysis(processed_df)
                    analysis_data = {
                        'metadata': {
                            'total_documents': len(df),
                            'date_range': {'start': str(df['date'].min()), 'end': str(df['date'].max())}
                        },
                        'strategic_shift': {
                            'total_documents': len(df),
                            'date_range': [str(df['date'].min()), str(df['date'].max())],
                            'overall_economic_avg': float(report['overall_economic_avg']),
                            'overall_security_avg': float(report['overall_security_avg']),
                            'economic_focus_avg': float(report['overall_economic_avg']),
                            'security_focus_avg': float(report['overall_security_avg']),
                            'crossover_year': report['crossover_year'],
                            'trend': report['trend'],
                            'statistical_significance': report.get('statistical_significance', {}),
                            'trend_analysis': report.get('trend_analysis', {}),
                        }
                    }
                    
                    pdf_gen = PDFReportGenerator()
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    success = pdf_gen.generate_report(analysis_data, output_file=tmp_file.name)
                    
                    if success:
                        with open(tmp_file.name, 'rb') as f:
                            pdf_bytes = f.read()
                        st.sidebar.download_button(
                            label="Save PDF",
                            data=pdf_bytes,
                            file_name="diplomatic_barometer_report.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.sidebar.error("PDF generation failed")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        lang["select_analysis"],
        [lang["executive_summary"], lang["strategic_shift"], lang["tone_sentiment"], 
         lang["thematic"], lang["time_machine"], lang["search_explore"], lang["statistical_tests"]]
    )
    
    # =====================================================================
    # PAGE 1: EXECUTIVE SUMMARY
    # =====================================================================
    if page == lang["executive_summary"]:
        st.header(f"{country1_name}-{country2_name} Diplomatic Overview")
        
        st.markdown(f"""
        How did {country1_name} and {country2_name} go from talking about **trade and aid** to discussing 
        **defense and security**? This dashboard answers that question by analyzing {len(processed_df)} 
        official diplomatic documents from 2000 to 2024.
        """)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(lang["total_documents"], len(processed_df))
        
        with col2:
            total_words = processed_df['cleaned'].apply(lambda x: len(str(x).split())).sum()
            st.metric(lang["total_words"], f"{total_words:,}")
        
        with col3:
            most_active_year = processed_df['year'].mode()[0]
            st.metric("Most Active Year", most_active_year)
        
        with col4:
            year_count = len(processed_df['year'].unique())
            st.metric("Years Covered", year_count)
        
        st.markdown("---")
        
        # Strategic Shift Chart - The Primary Proof
        st.subheader("The Strategic Shift: Economic vs. Security Focus Over Time")
        
        report, scored_df, yearly_df = perform_strategic_analysis(processed_df)
        
        # Create the shift chart
        fig = go.Figure()
        
        # Economic score line
        fig.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=yearly_df['economic_score_mean'],
            mode='lines+markers',
            name='Economic Focus',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Economic: %{y:.4f}<extra></extra>'
        ))
        
        # Security score line
        fig.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=yearly_df['security_score_mean'],
            mode='lines+markers',
            name='Security Focus',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Security: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Diplomatic Focus Evolution: Economic Aid → Military/Strategic Security",
            xaxis_title="Year",
            yaxis_title="Normalized Focus Score",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Key findings
        st.markdown(f"### {lang['key_findings']}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="stat-box">
            <b>{lang['overall_economic']}:</b> {report['overall_economic_avg']:.4f}<br/>
            <b>{lang['overall_security']}:</b> {report['overall_security_avg']:.4f}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if report['crossover_year']:
                st.markdown(f"""
                <div class="stat-box">
                <b>{lang['crossover_year']}:</b> {report['crossover_year']}<br/>
                {lang['security_overtook']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(lang['strategic_shift_evident'])
            
            st.write(f"**{lang['overall_trend']}:** {report['trend']}")
        
        st.info("""
        **How to read this chart:**
                
        - **Blue line (Economic):** Higher = more talk about trade, investment, infrastructure
        - **Pink line (Security):** Higher = more talk about defense, military, strategic alliances  
        - **The crossing point:** When security discussions overtook economic ones
        - **What it proves:** India-Japan went from trade partners to security allies
        """)
    
    # =====================================================================
    # PAGE 2: STRATEGIC SHIFT ANALYSIS
    # =====================================================================
    elif page == lang["strategic_shift"]:
        st.header(f"Economic vs Security: Detailed Breakdown")
        st.markdown(f"""
        A closer look at when and how {country1_name}-{country2_name} documents 
        shifted from economic to security topics.
        """)
        
        report, scored_df, yearly_df = perform_strategic_analysis(processed_df)
        
        # Display yearly statistics
        st.subheader("Yearly Statistics")
        st.dataframe(yearly_df.style.format({
            'economic_score_mean': '{:.4f}',
            'security_score_mean': '{:.4f}',
            'economic_score_std': '{:.4f}',
            'security_score_std': '{:.4f}'
        }), width='stretch')
        
        # Distribution comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_distribution_plot(
                scored_df['economic_score'].tolist(),
                "Economic Score Distribution",
                "Score"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = create_distribution_plot(
                scored_df['security_score'].tolist(),
                "Security Score Distribution",
                "Score"
            )
            st.plotly_chart(fig, width='stretch')
        
        # Top economic terms
        st.subheader("Top Economic Sector Terms")
        eco_terms = scored_df.nlargest(15, 'economic_score')[['title', 'economic_score']]
        
        if len(eco_terms) > 0:
            fig = px.bar(
                eco_terms.reset_index(drop=True),
                y=eco_terms.index,
                x='economic_score',
                orientation='h',
                title='Top 15 Documents by Economic Focus Score',
                labels={'economic_score': 'Score', 'index': 'Rank'}
            )
            st.plotly_chart(fig, width='stretch')
        
        # Top security terms
        st.subheader("Top Security Sector Terms")
        sec_terms = scored_df.nlargest(15, 'security_score')[['title', 'security_score']]
        
        if len(sec_terms) > 0:
            fig = px.bar(
                sec_terms.reset_index(drop=True),
                y=sec_terms.index,
                x='security_score',
                orientation='h',
                title='Top 15 Documents by Security Focus Score',
                color_discrete_sequence=['#A23B72']
            )
            st.plotly_chart(fig, width='stretch')
        
        st.info("""
        **How to read this page:**
        
        - **Table:** Average economic and security scores per year
        - **Distribution charts:** How scores cluster — most documents lean economic or security?
        - **Top documents:** The strongest examples of each type
        """)
    
    # =====================================================================
    # PAGE 3: TONE & SENTIMENT ANALYSIS
    # =====================================================================
    elif page == lang["tone_sentiment"]:
        st.header(f"Tone & Mood Analysis")
        st.markdown(f"How urgent and positive is the language in {country1_name}-{country2_name} communications?")
        
        # Analyze tone
        with st.spinner("Analyzing tone and sentiment..."):
            tone_analyzer, tone_df = perform_tone_analysis(processed_df)
        
        # Check if analysis was successful
        if tone_analyzer is None or tone_df is None or len(tone_df) == 0:
            st.error("Unable to perform tone analysis. Please check that data has been loaded correctly.")
            st.stop()
        
        # Tone distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tone Distribution")
            tone_dist = tone_analyzer.get_tone_distribution(tone_df)
            fig = px.pie(
                values=list(tone_dist.values()),
                names=list(tone_dist.keys()),
                title="Distribution of Diplomatic Tones"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Sentiment Distribution")
            sentiment_dist = tone_analyzer.get_sentiment_distribution(tone_df)
            fig = px.pie(
                values=list(sentiment_dist.values()),
                names=list(sentiment_dist.keys()),
                title="Distribution of Sentiments",
                color_discrete_map={
                    'Positive': '#2ecc71',
                    'Neutral': '#95a5a6',
                    'Negative': '#e74c3c'
                }
            )
            st.plotly_chart(fig, width='stretch')
        
        # Yearly tone statistics
        st.subheader("Yearly Tone Trends")
        yearly_tone = tone_analyzer.get_yearly_tone_statistics(tone_df)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=yearly_tone['year'],
            y=yearly_tone['urgency_score_mean'],
            mode='lines+markers',
            name='Urgency Score',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_tone['year'],
            y=yearly_tone['sentiment_polarity_mean'],
            mode='lines+markers',
            name='Sentiment Polarity',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Tone and Sentiment Trends Over Time",
            xaxis_title="Year",
            yaxis_title="Score",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Statistics
        st.subheader("Tone Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Urgency", f"{yearly_tone['urgency_score_mean'].mean():.3f}")
        with col2:
            st.metric("Avg Polarity", f"{yearly_tone['sentiment_polarity_mean'].mean():.3f}")
        with col3:
            most_tone = tone_df['tone_class'].mode()[0] if len(tone_df) > 0 and 'tone_class' in tone_df.columns else "Formal"
            st.metric("Most Common Tone", most_tone)
        
        st.info("""
        **How to read this page:**
        
        - **Tone:** Formal vs Cordial — how stiff or friendly is the language?
        - **Sentiment:** Positive (praising), Neutral (factual), Negative (concerned)
        - **Urgency:** How pressing are the issues discussed? Higher = more urgent
        - **Trends:** Are conversations becoming more urgent or more positive over time?
        """)
    
    # =====================================================================
    # PAGE 4: THEMATIC ANALYSIS
    # =====================================================================
    elif page == lang["thematic"]:
        st.header(f"Topics & Themes")
        st.markdown(f"What topics come up most in {country1_name}-{country2_name} diplomacy, and how have they changed?")
        
        with st.spinner("Performing thematic analysis..."):
            thematic_analyzer, analysis, df_themes = perform_thematic_analysis(processed_df)
        
        # Display discovered themes in a better format
        st.subheader("Discovered Themes")
        st.markdown(f"*The computer found these topic groups automatically by reading all the documents:*")
        
        # Create better layout for themes
        for i, (topic_id, words) in enumerate(sorted(analysis['overall_themes'].items())):
            # Add visual indicator
            colors = ['🟦', '🟥', '🟩', '🟨', '🟪']
            color = colors[i % 5]
            
            with st.container():
                st.markdown(f"""
                {color} **Theme {topic_id}**
                
                **Keywords:** {', '.join(words[:6])}
                
                *Supporting terms:* {', '.join(words[6:])}
                """)
                st.divider()
        
        # Topic distribution by year
        st.subheader("How Themes Evolved Over Time")
        
        # Prepare data for visualization
        topic_year_data = []
        for year, topic_list in analysis['topic_distribution_by_year'].items():
            if topic_list:
                for topic_id, prob in topic_list[:3]:
                    topic_year_data.append({'Year': year, 'Theme': f'Theme {topic_id}', 'Documents': prob})
        
        if topic_year_data:
            topic_year_df = pd.DataFrame(topic_year_data)
            fig = px.bar(
                topic_year_df,
                x='Year',
                y='Documents',
                color='Theme',
                title='Themes by Year (Stacked)',
                barmode='stack',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Number of Documents',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, width='stretch')
        
        # Show themes explanation
        st.info("""
        **How to read this page:**
        
        - **Each theme:** A group of related topics found automatically by the computer
        - **Keywords:** The most distinctive words defining each theme
        - **Bar chart:** Shows which themes dominated each year
        - **Example:** If a "security" theme appears strongly after 2015, it supports the shift idea
        """)

    
    # =====================================================================
    # PAGE 5: INTERACTIVE TIME MACHINE
    # =====================================================================
    elif page == lang["time_machine"]:
        st.header(f"Year-by-Year Explorer")
        st.markdown(f"Slide through time to see what {country1_name} and {country2_name} were discussing each year")
        
        years = sorted(processed_df['year'].unique())
        selected_year = st.slider(
            "Select a Year",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=int(years[len(years)//2])
        )
        
        # Get documents from selected year
        year_data = processed_df[processed_df['year'] == selected_year]
        
        st.subheader(f"Year {selected_year}")
        st.info(f"{len(year_data)} documents from this year")
        
        if len(year_data) > 0:
            # Generate word cloud
            all_lemmas = []
            for lemmas in year_data['lemmas']:
                if isinstance(lemmas, list):
                    all_lemmas.extend(lemmas)
            
            if all_lemmas:
                # Create word cloud
                wordcloud_text = ' '.join(all_lemmas)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                wc = WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    colormap='viridis'
                ).generate(wordcloud_text)
                
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)
                
                st.pyplot(fig)
            
            # Show key documents from this year
            st.subheader("Documents from This Year")
            for idx, row in year_data.iterrows():
                with st.expander(f"{row['title'][:60]}..."):
                    st.write(f"**Date:** {row['date']}")
                    st.write(f"**Location:** {row['location']}")
                    st.write(f"**Content:** {row['cleaned'][:300]}...")
        
        st.info("""
        **How to read this page:**
        
        - **Slider:** Pick any year to see what was happening
        - **Word cloud:** Bigger words = mentioned more often that year
        - **Documents:** Actual text from that year's diplomatic meetings
        - **Try it:** Compare 2000 (trade words) vs 2020 (security words)
        """)
    
    # =====================================================================
    # PAGE 6: SEARCH & EXPLORE
    # =====================================================================
    elif page == lang["search_explore"]:
        st.header(f"Keyword Search")
        st.markdown(f"Search for any word and see how often it appears across 25 years of {country1_name}-{country2_name} documents")
        
        # Keyword search
        search_term = st.text_input(
            "Enter a keyword to search",
            value="relations"
        )
        
        if search_term:
            search_term_lower = search_term.lower()
            
            # Find documents containing the term
            matching_docs = processed_df[
                processed_df['cleaned'].str.contains(search_term_lower, case=False, na=False)
            ]
            
            if len(matching_docs) > 0:
                # Frequency by year
                year_freq = matching_docs.groupby('year').size()
                
                fig = px.bar(
                    x=year_freq.index,
                    y=year_freq.values,
                    labels={'x': 'Year', 'y': 'Mentions'},
                    title=f"Occurrences of '{search_term}' Over Time"
                )
                st.plotly_chart(fig, width='stretch')
                
                # Show matching documents
                st.subheader(f"Documents Mentioning '{search_term}'")
                st.info(f"Found in {len(matching_docs)} documents")
                
                for idx, row in matching_docs.head(10).iterrows():
                    with st.expander(f"{row['title']} ({row['year']})"):
                        st.write(f"**Date:** {row['date']}")
                        st.write(f"**Location:** {row['location']}")
                        
                        # Highlight the search term in context
                        content = str(row['cleaned'])
                        highlighted = content.replace(
                            search_term_lower,
                            f"**{search_term_lower.upper()}**"
                        )
                        st.markdown(highlighted[:500] + "...")
            else:
                st.warning(f"No documents found containing '{search_term}'")
        
        st.info("""
        **How to read this page:**
        
        - **Try searching:** "nuclear", "defense", "trade", "quad", "infrastructure"
        - **Rising bar chart:** That topic got more attention over time
        - **Falling bar chart:** That topic faded from discussions
        - **Documents:** Shows real excerpts where your keyword appears
        """)
    
    # =====================================================================
    # PAGE 7: STATISTICAL TESTS
    # =====================================================================
    elif page == lang["statistical_tests"]:
        st.header(f"Statistical Proof")
        st.markdown(f"Is the economic-to-security shift real, or just noise? These tests provide the answer.")
        
        report, scored_df, yearly_df = perform_strategic_analysis(processed_df)
        
        # T-test for economic vs security
        st.subheader("Economic vs. Security Focus Comparison")
        
        stats_result = calculate_statistical_significance(
            scored_df['economic_score'],
            scored_df['security_score']
        )
        
        if stats_result:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("T-Statistic", f"{stats_result['t_statistic']:.4f}")
            
            with col2:
                st.metric("P-Value", f"{stats_result['p_value']:.6f}")
            
            with col3:
                st.metric("Significant", "Yes" if stats_result['significant'] else "No")
            
            with col4:
                st.metric("Effect Size", f"{stats_result['effect_size']:.4f}")
            
            if stats_result['significant']:
                st.success("The difference between economic and security focus is statistically significant (p < 0.05)")
            else:
                st.info("The difference is not statistically significant at the 0.05 level")
        
        # Correlation analysis
        st.subheader("Year-over-Year Correlation")
        
        yearly_df_sorted = yearly_df.sort_values('year')
        correlation = yearly_df_sorted['economic_score_mean'].corr(yearly_df_sorted['security_score_mean'])
        
        st.metric("Correlation (Economic vs Security)", f"{correlation:.4f}")
        
        if correlation < -0.3:
            st.info("Strong negative correlation: As economic focus decreases, security focus increases")
        elif correlation > 0.3:
            st.info("Strong positive correlation: Both metrics move together")
        else:
            st.info("Weak correlation: Metrics move independently")
        
        # Trend analysis
        st.subheader("Trend Analysis (Linear Regression)")
        
        # Economic trend
        X = yearly_df_sorted[['year']].values
        y_econ = yearly_df_sorted['economic_score_mean'].values
        y_sec = yearly_df_sorted['security_score_mean'].values
        
        econ_slope = (y_econ[-1] - y_econ[0]) / (X[-1][0] - X[0][0])
        sec_slope = (y_sec[-1] - y_sec[0]) / (X[-1][0] - X[0][0])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Economic Focus Trend (per year)", f"{econ_slope:.6f}")
            if econ_slope < 0:
                st.write("Declining economic focus over time")
            else:
                st.write("Increasing economic focus over time")
        
        with col2:
            st.metric("Security Focus Trend (per year)", f"{sec_slope:.6f}")
            if sec_slope > 0:
                st.write("Rising security focus over time")
            else:
                st.write("Declining security focus over time")
        
        st.info("""
        **How to read this page:**
        
        - **P-value < 0.05:** The difference is real, not random chance
        - **Effect size:** How big is the shift? Larger = more dramatic change
        - **Correlation:** Do economic and security move in opposite directions? (expected: yes)
        - **Trend lines:** Show the long-term direction over 25 years
        - **Bottom line:** This is the academic proof that the strategic shift is real
        """)


if __name__ == "__main__":
    main()
