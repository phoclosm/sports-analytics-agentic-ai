import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import rankdata
import numpy as np

OUTPUT_DIR = "outputfoot"

st.set_page_config(layout="wide", page_title="⚽ Football Analysis Dashboard", page_icon="⚽")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_latest_analysis_json():
    """Finds and loads the latest JSON file from outputfoot directory."""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return None, "❌ Output directory 'outputfoot' not found."

    json_files = glob.glob(str(output_path / "*.json"))
    
    if not json_files:
        return None, "❌ No JSON files found in 'outputfoot' directory."

    latest_file = max(json_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, f"✅ Loaded: {Path(latest_file).name}"
    except Exception as e:
        return None, f"❌ Error: {e}"

def create_player_dataframe(data):
    """Converts player data into DataFrame."""
    if not data or 'player_performances' not in data:
        return pd.DataFrame()
    
    rows = []
    for player in data['player_performances']:
        row = {
            'player_id': player.get('player_id', 'Unknown'),
            'player_name': player.get('player_name', 'Unknown'),
            'team': player.get('team', 'Unknown'),
            'position': player.get('position', 'Unknown'),
            'estimated_minutes': player.get('detection_summary', {}).get('estimated_minutes', 0),
        }
        
        # Get aggregated stats
        agg_stats = player.get('aggregated_statistics', {})
        row['total_goals'] = agg_stats.get('total_goals', 0)
        row['total_assists'] = agg_stats.get('total_assists', 0)
        row['total_shots'] = agg_stats.get('total_shots', 0)
        row['overall_rating'] = agg_stats.get('overall_rating', 0)
        
        # Get pressure performance from detailed analysis
        detailed = player.get('detailed_performance_analysis', [])
        if detailed:
            # Average pressure metrics across all segments
            pressure_ratings = []
            for segment in detailed:
                pressure = segment.get('pressure_performance', {})
                if 'pressure_composure_rating' in pressure:
                    pressure_ratings.append(pressure['pressure_composure_rating'])
            
            row['pressure_composure'] = np.mean(pressure_ratings) if pressure_ratings else 0
            row['clutch_moments'] = sum(len(seg.get('pressure_performance', {}).get('clutch_moments', [])) for seg in detailed)
        else:
            row['pressure_composure'] = 0
            row['clutch_moments'] = 0
        
        # Get contextual performance
        context = player.get('contextual_performance', {})
        row['pressure_performance'] = float(context.get('performance_in_pressure_moments', '0/10').split('/')[0])
        
        rows.append(row)
    
    df = pd.DataFrame(rows).fillna(0)
    
    # Calculate pass accuracy if available from detailed analysis
    df['pass_accuracy'] = 0
    for idx, player in enumerate(data['player_performances']):
        detailed = player.get('detailed_performance_analysis', [])
        if detailed:
            accuracies = []
            for segment in detailed:
                stats = segment.get('key_stats', {})
                acc = stats.get('pass_accuracy_pct', 0)
                if acc > 0:
                    accuracies.append(acc)
            if accuracies:
                df.loc[idx, 'pass_accuracy'] = np.mean(accuracies)
    
    return df

def generate_overall_rankings(df):
    """Generate overall performance rankings."""
    if df.empty:
        return df
    
    stats_to_rank = {
        'total_goals': 'Overall_Goals',
        'total_assists': 'Overall_Assists',
        'total_shots': 'Overall_Shots',
        'pass_accuracy': 'Overall_Pass',
        'overall_rating': 'Overall_Rating'
    }
    
    rank_df = pd.DataFrame(index=df.index)
    
    for stat, label in stats_to_rank.items():
        if stat in df.columns and df[stat].sum() > 0:
            ranks = rankdata(-df[stat].fillna(0), method='min')
            percentile = (len(df) - ranks + 1) / len(df)
            rank_df[f'{label}_score'] = (percentile * 100).round(0)
        else:
            rank_df[f'{label}_score'] = 50
    
    score_cols = [col for col in rank_df.columns if '_score' in col]
    if score_cols:
        rank_df['OVERALL_SCORE'] = rank_df[score_cols].mean(axis=1).round(0)
        rank_df['OVERALL_RANK'] = rankdata(-rank_df['OVERALL_SCORE'], method='min').astype(int)
    
    return df.join(rank_df)

def generate_pressure_rankings(df):
    """Generate pressure-specific performance rankings."""
    if df.empty:
        return df
    
    pressure_stats = {
        'pressure_performance': 'Pressure_Perf',
        'pressure_composure': 'Pressure_Composure',
        'clutch_moments': 'Pressure_Clutch',
        'total_goals': 'Pressure_Goals',
        'overall_rating': 'Pressure_Rating'
    }
    
    pressure_df = pd.DataFrame(index=df.index)
    
    for stat, label in pressure_stats.items():
        if stat in df.columns and df[stat].sum() > 0:
            ranks = rankdata(-df[stat].fillna(0), method='min')
            percentile = (len(df) - ranks + 1) / len(df)
            pressure_df[f'{label}_score'] = (percentile * 100).round(0)
        else:
            pressure_df[f'{label}_score'] = 50
    
    score_cols = [col for col in pressure_df.columns if '_score' in col]
    if score_cols:
        pressure_df['PRESSURE_SCORE'] = pressure_df[score_cols].mean(axis=1).round(0)
        pressure_df['PRESSURE_RANK'] = rankdata(-pressure_df['PRESSURE_SCORE'], method='min').astype(int)
    
    return df.join(pressure_df)

def display_match_overview(data, df):
    """Display match overview with enhanced visualizations."""
    st.header("⚽ Match Overview")
    
    match_info = data.get('match_information', {})
    video_struct = match_info.get('video_structure', {})
    match_events = match_info.get('match_events', {})
    
    # Score display
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"### {video_struct.get('team_home', 'Team A')}")
        st.metric("", video_struct.get('final_score', {}).get('home', 0), label_visibility="collapsed")
    
    with col2:
        st.markdown("### VS")
        st.markdown(f"**{video_struct.get('competition', 'Competition')}**")
        st.caption(f"📍 {video_struct.get('venue', 'Stadium')}")
    
    with col3:
        st.markdown(f"### {video_struct.get('team_away', 'Team B')}")
        st.metric("", video_struct.get('final_score', {}).get('away', 0), label_visibility="collapsed")
    
    st.divider()
    
    # Match Statistics
    st.subheader("📊 Match Statistics")
    stats_summary = data.get('statistical_summary', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Goals", stats_summary.get('total_goals', 0), help="Goals scored by both teams")
    col2.metric("Yellow Cards", stats_summary.get('yellow_cards', 0))
    col3.metric("Red Cards", stats_summary.get('red_cards', 0))
    col4.metric("Pressure Moments", stats_summary.get('pressure_moments_count', 0))
    col5.metric("Set Pieces", stats_summary.get('set_pieces_count', 0))
    
    # Goals Timeline Visualization
    st.subheader("⚽ Goals Timeline")
    goals = match_events.get('goals_scored', [])
    
    if goals:
        # Create timeline chart
        timeline_data = []
        for goal in goals:
            minute = goal.get('game_minute', '0')
            minute_num = int(''.join(filter(str.isdigit, minute.split("'")[0])))
            timeline_data.append({
                'Minute': minute_num,
                'Scorer': goal.get('scorer_name', 'Unknown'),
                'Team': goal.get('scoring_team', 'Unknown'),
                'Type': goal.get('goal_type', 'open_play'),
                'Score': f"{goal.get('score_after', {}).get('home', 0)}-{goal.get('score_after', {}).get('away', 0)}"
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig = px.scatter(timeline_df, x='Minute', y='Team', 
                        size=[20]*len(timeline_df),
                        color='Team',
                        hover_data=['Scorer', 'Type', 'Score'],
                        title='Goal Distribution Timeline',
                        height=300)
        fig.update_layout(xaxis_range=[0, 95])
        st.plotly_chart(fig, use_container_width=True)
        
        # Goals table
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    
    # Score progression chart
    st.subheader("📈 Score Progression")
    if goals:
        progression_data = [{'Minute': 0, 'Home': 0, 'Away': 0}]
        for goal in goals:
            minute = int(''.join(filter(str.isdigit, goal.get('game_minute', '0').split("'")[0])))
            score_after = goal.get('score_after', {})
            progression_data.append({
                'Minute': minute,
                'Home': score_after.get('home', 0),
                'Away': score_after.get('away', 0)
            })
        
        prog_df = pd.DataFrame(progression_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prog_df['Minute'], y=prog_df['Home'], 
                                mode='lines+markers', name=video_struct.get('team_home', 'Home'),
                                line=dict(color='#3b82f6', width=3)))
        fig.add_trace(go.Scatter(x=prog_df['Minute'], y=prog_df['Away'], 
                                mode='lines+markers', name=video_struct.get('team_away', 'Away'),
                                line=dict(color='#ef4444', width=3)))
        fig.update_layout(title='Score Over Time', xaxis_title='Minute', yaxis_title='Goals',
                         height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

def display_overall_leaderboard(df):
    """Display overall performance leaderboard."""
    st.header("🏆 Overall Performance Leaderboard")
    
    if df.empty:
        st.warning("No player data available.")
        return
    
    # Add medal emojis
    def add_medal(rank):
        if rank == 1:
            return '🥇'
        elif rank == 2:
            return '🥈'
        elif rank == 3:
            return '🥉'
        return ''
    
    display_df = df[['OVERALL_RANK', 'player_name', 'team', 'position', 'OVERALL_SCORE',
                     'total_goals', 'total_assists', 'total_shots', 'pass_accuracy', 
                     'overall_rating']].copy()
    
    display_df['Medal'] = display_df['OVERALL_RANK'].apply(add_medal)
    display_df = display_df.rename(columns={
        'OVERALL_RANK': 'Rank',
        'player_name': 'Player',
        'team': 'Team',
        'position': 'Position',
        'OVERALL_SCORE': 'Score',
        'total_goals': 'Goals',
        'total_assists': 'Assists',
        'total_shots': 'Shots',
        'pass_accuracy': 'Pass %',
        'overall_rating': 'Rating'
    })
    
    # Reorder columns
    display_df = display_df[['Medal', 'Rank', 'Player', 'Team', 'Position', 'Score',
                             'Goals', 'Assists', 'Shots', 'Pass %', 'Rating']]
    
    st.dataframe(display_df.sort_values('Rank'), use_container_width=True, hide_index=True)
    
    # Top 5 visualization
    st.subheader("🌟 Top 5 Players - Performance Breakdown")
    top5 = df.nsmallest(5, 'OVERALL_RANK')
    
    radar_data = []
    for _, player in top5.iterrows():
        radar_data.append({
            'Player': player['player_name'],
            'Goals': player.get('Overall_Goals_score', 50),
            'Assists': player.get('Overall_Assists_score', 50),
            'Shots': player.get('Overall_Shots_score', 50),
            'Pass%': player.get('Overall_Pass_score', 50),
            'Rating': player.get('Overall_Rating_score', 50)
        })
    
    # Create multi-player radar chart
    fig = go.Figure()
    
    categories = ['Goals', 'Assists', 'Shots', 'Pass%', 'Rating']
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
    
    for idx, player_data in enumerate(radar_data):
        values = [player_data[cat] for cat in categories]
        values.append(values[0])  # Close the radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=player_data['Player'],
            line=dict(color=colors[idx % len(colors)], width=2),
            fill='toself',
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def display_pressure_leaderboard(df):
    """Display pressure-specific performance leaderboard."""
    st.header("🔥 Pressure Performance Leaderboard")
    
    if df.empty:
        st.warning("No player data available.")
        return
    
    st.info("📌 This leaderboard ranks players based on their performance under pressure situations, composure, and clutch moments.")
    
    # Add icons
    def add_icon(rank):
        if rank == 1:
            return '🔥'
        elif rank == 2:
            return '⚡'
        elif rank == 3:
            return '💪'
        return ''
    
    display_df = df[['PRESSURE_RANK', 'player_name', 'team', 'position', 'PRESSURE_SCORE',
                     'pressure_performance', 'pressure_composure', 'clutch_moments',
                     'total_goals']].copy()
    
    display_df['Icon'] = display_df['PRESSURE_RANK'].apply(add_icon)
    display_df = display_df.rename(columns={
        'PRESSURE_RANK': 'Rank',
        'player_name': 'Player',
        'team': 'Team',
        'position': 'Position',
        'PRESSURE_SCORE': 'Pressure Score',
        'pressure_performance': 'Pressure Perf',
        'pressure_composure': 'Composure',
        'clutch_moments': 'Clutch Moments',
        'total_goals': 'Goals'
    })
    
    display_df = display_df[['Icon', 'Rank', 'Player', 'Team', 'Position', 'Pressure Score',
                             'Pressure Perf', 'Composure', 'Clutch Moments', 'Goals']]
    
    st.dataframe(display_df.sort_values('Rank'), use_container_width=True, hide_index=True)
    
    # Pressure vs Performance scatter
    st.subheader("💎 Pressure Performance vs Overall Rating")
    
    fig = px.scatter(df, 
                     x='pressure_performance', 
                     y='overall_rating',
                     size='clutch_moments',
                     color='team',
                     hover_name='player_name',
                     hover_data=['total_goals', 'pressure_composure'],
                     title='Players Under Pressure',
                     labels={'pressure_performance': 'Pressure Performance (0-10)',
                            'overall_rating': 'Overall Rating (0-10)'},
                     height=500)
    
    fig.update_layout(xaxis_range=[0, 11], yaxis_range=[0, 11])
    st.plotly_chart(fig, use_container_width=True)
    
    # Clutch performers bar chart
    st.subheader("🎯 Clutch Performers")
    clutch_df = df[df['clutch_moments'] > 0].nlargest(10, 'clutch_moments')
    
    if not clutch_df.empty:
        fig = px.bar(clutch_df.sort_values('clutch_moments', ascending=True),
                     y='player_name',
                     x='clutch_moments',
                     color='pressure_composure',
                     orientation='h',
                     title='Players with Most Clutch Moments',
                     labels={'clutch_moments': 'Clutch Moments', 'player_name': 'Player'},
                     color_continuous_scale='RdYlGn',
                     height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No clutch moments recorded in this match.")

def display_player_reports(df, data):
    """Display detailed player reports with enhanced visualizations."""
    st.header("👤 Player Reports")
    
    if df.empty:
        st.warning("No player data available.")
        return
    
    player_list = df['player_name'].unique().tolist()
    selected_player = st.selectbox("Select a Player", player_list, key='player_report')
    
    if selected_player:
        player_row = df[df['player_name'] == selected_player].iloc[0]
        
        # Get detailed analysis
        player_detailed = None
        for p in data.get('player_performances', []):
            if p.get('player_name') == selected_player:
                player_detailed = p
                break
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"📋 {selected_player}")
            st.markdown(f"**Team:** {player_row['team']}")
            st.markdown(f"**Position:** {player_row['position']}")
            st.markdown(f"**Minutes:** {player_row['estimated_minutes']}")
            
            st.divider()
            
            # Performance badges
            overall_rank = int(player_row.get('OVERALL_RANK', 0))
            pressure_rank = int(player_row.get('PRESSURE_RANK', 0))
            
            if overall_rank <= 3:
                st.success(f"🏆 Overall Rank: #{overall_rank}")
            else:
                st.info(f"📊 Overall Rank: #{overall_rank}")
            
            if pressure_rank <= 3:
                st.success(f"🔥 Pressure Rank: #{pressure_rank}")
            else:
                st.info(f"💪 Pressure Rank: #{pressure_rank}")
            
            st.divider()
            
            # Key metrics
            st.metric("Goals", f"{player_row.get('total_goals', 0):.0f}")
            st.metric("Assists", f"{player_row.get('total_assists', 0):.0f}")
            st.metric("Shots", f"{player_row.get('total_shots', 0):.0f}")
            st.metric("Pass Accuracy", f"{player_row.get('pass_accuracy', 0):.1f}%")
            st.metric("Overall Rating", f"{player_row.get('overall_rating', 0):.1f}/10")
            st.metric("Pressure Performance", f"{player_row.get('pressure_performance', 0):.1f}/10")
            st.metric("Clutch Moments", f"{player_row.get('clutch_moments', 0):.0f}")
        
        with col2:
            # Performance radar
            st.subheader("📊 Performance Profile")
            
            radar_categories = ['Goals', 'Assists', 'Shots', 'Pass%', 'Rating']
            radar_values = [
                player_row.get('Overall_Goals_score', 50),
                player_row.get('Overall_Assists_score', 50),
                player_row.get('Overall_Shots_score', 50),
                player_row.get('Overall_Pass_score', 50),
                player_row.get('Overall_Rating_score', 50)
            ]
            radar_values.append(radar_values[0])
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=radar_categories + [radar_categories[0]],
                fill='toself',
                name=selected_player,
                line=dict(color='#3b82f6', width=3),
                fillcolor='rgba(59, 130, 246, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Pressure metrics
            st.subheader("🔥 Pressure Metrics")
            
            pressure_metrics = {
                'Pressure Performance': player_row.get('pressure_performance', 0),
                'Composure': player_row.get('pressure_composure', 0),
                'Overall Rating': player_row.get('overall_rating', 0)
            }
            
            fig = go.Figure(go.Bar(
                x=list(pressure_metrics.values()),
                y=list(pressure_metrics.keys()),
                orientation='h',
                marker=dict(color=['#ef4444', '#f59e0b', '#10b981'])
            ))
            fig.update_layout(xaxis_range=[0, 10], height=250, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Segment performance
            if player_detailed and player_detailed.get('detailed_performance_analysis'):
                st.subheader("⏱️ Performance Timeline")
                
                segments = player_detailed['detailed_performance_analysis']
                segment_data = []
                
                for seg in segments:
                    timeframe = seg.get('timeframe', '')
                    game_minute = seg.get('game_minute_range', '')
                    impact = seg.get('match_impact', {})
                    
                    segment_data.append({
                        'Timeframe': game_minute,
                        'Rating': impact.get('overall_performance_rating', 0),
                        'Impact': impact.get('impact_on_scoreline', 'neutral')
                    })
                
                if segment_data:
                    seg_df = pd.DataFrame(segment_data)
                    st.dataframe(seg_df, use_container_width=True, hide_index=True)

def display_head_to_head(df):
    """Display head-to-head comparison for all players."""
    st.header("🆚 Head-to-Head Player Comparison")
    
    if df.empty or len(df) < 2:
        st.warning("Need at least 2 players for comparison.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox("Select Player 1", df['player_name'].tolist(), key='h2h_p1')
    
    with col2:
        player2 = st.selectbox("Select Player 2", df['player_name'].tolist(), key='h2h_p2')
    
    if player1 and player2 and player1 != player2:
        p1_data = df[df['player_name'] == player1].iloc[0]
        p2_data = df[df['player_name'] == player2].iloc[0]
        
        st.subheader(f"⚔️ {player1} vs {player2}")
        
        # Overall comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"### {player1}")
            st.markdown(f"**{p1_data['team']}** | {p1_data['position']}")
            st.metric("Overall Rank", f"#{int(p1_data.get('OVERALL_RANK', 0))}")
            st.metric("Pressure Rank", f"#{int(p1_data.get('PRESSURE_RANK', 0))}")
        
        with col2:
            st.markdown("### 📊")
            st.markdown("**Comparison**")
            
        with col3:
            st.markdown(f"### {player2}")
            st.markdown(f"**{p2_data['team']}** | {p2_data['position']}")
            st.metric("Overall Rank", f"#{int(p2_data.get('OVERALL_RANK', 0))}")
            st.metric("Pressure Rank", f"#{int(p2_data.get('PRESSURE_RANK', 0))}")
        
        st.divider()
        
        # Stats comparison table
        st.subheader("📈 Statistical Comparison")
        
        comp_df = pd.DataFrame({
            'Metric': ['Goals', 'Assists', 'Shots', 'Pass Accuracy %', 'Overall Rating',
                      'Pressure Performance', 'Composure', 'Clutch Moments'],
            player1: [
                p1_data.get('total_goals', 0),
                p1_data.get('total_assists', 0),
                p1_data.get('total_shots', 0),
                round(p1_data.get('pass_accuracy', 0), 1),
                round(p1_data.get('overall_rating', 0), 1),
                round(p1_data.get('pressure_performance', 0), 1),
                round(p1_data.get('pressure_composure', 0), 1),
                p1_data.get('clutch_moments', 0)
            ],
            player2: [
                p2_data.get('total_goals', 0),
                p2_data.get('total_assists', 0),
                p2_data.get('total_shots', 0),
                round(p2_data.get('pass_accuracy', 0), 1),
                round(p2_data.get('overall_rating', 0), 1),
                round(p2_data.get('pressure_performance', 0), 1),
                round(p2_data.get('pressure_composure', 0), 1),
                p2_data.get('clutch_moments', 0)
            ]
        })
        
        st.dataframe(comp_df.set_index('Metric'), use_container_width=True)
        
        # Visual comparison
        st.subheader("📊 Visual Comparison")
        
        # Side-by-side radar charts
        col1, col2 = st.columns(2)
        
        categories = ['Goals', 'Assists', 'Shots', 'Pass%', 'Rating']
        
        with col1:
            p1_values = [
                p1_data.get('Overall_Goals_score', 50),
                p1_data.get('Overall_Assists_score', 50),
                p1_data.get('Overall_Shots_score', 50),
                p1_data.get('Overall_Pass_score', 50),
                p1_data.get('Overall_Rating_score', 50)
            ]
            p1_values.append(p1_values[0])
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatterpolar(
                r=p1_values,
                theta=categories + [categories[0]],
                fill='toself',
                name=player1,
                line=dict(color='#3b82f6', width=2),
                fillcolor='rgba(59, 130, 246, 0.3)'
            ))
            fig1.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=350,
                title=f"{player1}"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            p2_values = [
                p2_data.get('Overall_Goals_score', 50),
                p2_data.get('Overall_Assists_score', 50),
                p2_data.get('Overall_Shots_score', 50),
                p2_data.get('Overall_Pass_score', 50),
                p2_data.get('Overall_Rating_score', 50)
            ]
            p2_values.append(p2_values[0])
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=p2_values,
                theta=categories + [categories[0]],
                fill='toself',
                name=player2,
                line=dict(color='#ef4444', width=2),
                fillcolor='rgba(239, 68, 68, 0.3)'
            ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=350,
                title=f"{player2}"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Bar chart comparison
        st.subheader("📊 Direct Comparison")
        
        plot_df = comp_df.melt('Metric', var_name='Player', value_name='Value')
        
        fig = px.bar(plot_df, x='Metric', y='Value', color='Player',
                     barmode='group',
                     color_discrete_map={player1: '#3b82f6', player2: '#ef4444'},
                     height=450)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Winner determination
        st.subheader("🏆 Head-to-Head Winner")
        
        p1_wins = 0
        p2_wins = 0
        
        for idx, row in comp_df.iterrows():
            if row[player1] > row[player2]:
                p1_wins += 1
            elif row[player2] > row[player1]:
                p2_wins += 1
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.metric(player1, f"{p1_wins} categories won")
        
        with col2:
            if p1_wins > p2_wins:
                st.success("👈 Winner!")
            elif p2_wins > p1_wins:
                st.success("Winner! 👉")
            else:
                st.info("🤝 Tie!")
        
        with col3:
            st.metric(player2, f"{p2_wins} categories won")

def display_mental_pressure_analysis(df, data):
    """Display comprehensive mental and pressure analysis."""
    st.header("🧠 Mental & Pressure Analysis")
    
    if df.empty:
        st.warning("No player data available.")
        return
    
    st.info("💡 This section analyzes how players perform under high-pressure situations and their mental resilience.")
    
    # Pressure intensity overview
    st.subheader("🔥 Pressure Intensity Overview")
    
    match_events = data.get('match_information', {}).get('match_events', {})
    pressure_moments = match_events.get('pressure_moments', [])
    
    if pressure_moments:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Pressure Moments", len(pressure_moments))
        
        with col2:
            avg_intensity = np.mean([pm.get('intensity_rating', 0) for pm in pressure_moments])
            st.metric("Average Intensity", f"{avg_intensity:.1f}/10")
        
        with col3:
            max_intensity = max([pm.get('intensity_rating', 0) for pm in pressure_moments])
            st.metric("Peak Intensity", f"{max_intensity}/10")
        
        # Pressure moments timeline
        st.markdown("#### ⏱️ Pressure Moments Timeline")
        
        pm_data = []
        for pm in pressure_moments:
            pm_data.append({
                'Time': pm.get('game_minute', 'Unknown'),
                'Type': pm.get('pressure_type', 'Unknown'),
                'Intensity': pm.get('intensity_rating', 0),
                'Team': pm.get('team_under_pressure', 'Unknown'),
                'Outcome': pm.get('outcome', 'Unknown')
            })
        
        pm_df = pd.DataFrame(pm_data)
        st.dataframe(pm_df, use_container_width=True, hide_index=True)
        
        # Pressure intensity visualization
        fig = px.bar(pm_df, x='Time', y='Intensity', color='Team',
                     title='Pressure Intensity Throughout Match',
                     labels={'Intensity': 'Pressure Intensity (0-10)'},
                     height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Player pressure performance analysis
    st.subheader("👥 Player Pressure Performance")
    
    # Scatter plot: Pressure Performance vs Overall Rating
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df,
                         x='pressure_performance',
                         y='overall_rating',
                         size='clutch_moments',
                         color='team',
                         hover_name='player_name',
                         hover_data=['total_goals', 'total_assists', 'pressure_composure'],
                         title='Pressure Performance vs Overall Rating',
                         labels={'pressure_performance': 'Pressure Performance (0-10)',
                                'overall_rating': 'Overall Rating (0-10)'},
                         height=400)
        fig.update_layout(xaxis_range=[0, 11], yaxis_range=[0, 11])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df,
                         x='pressure_composure',
                         y='total_goals',
                         size='clutch_moments',
                         color='team',
                         hover_name='player_name',
                         hover_data=['pressure_performance', 'overall_rating'],
                         title='Composure vs Goal Scoring',
                         labels={'pressure_composure': 'Composure (0-10)',
                                'total_goals': 'Goals Scored'},
                         height=400)
        fig.update_layout(xaxis_range=[0, 11])
        st.plotly_chart(fig, use_container_width=True)
    
    # Top pressure performers
    st.subheader("🌟 Top Pressure Performers")
    
    top_pressure = df.nlargest(5, 'pressure_performance')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Pressure Performance',
        x=top_pressure['player_name'],
        y=top_pressure['pressure_performance'],
        marker_color='#ef4444'
    ))
    
    fig.add_trace(go.Bar(
        name='Composure',
        x=top_pressure['player_name'],
        y=top_pressure['pressure_composure'],
        marker_color='#f59e0b'
    ))
    
    fig.update_layout(
        title='Top 5 Players - Pressure Metrics',
        xaxis_title='Player',
        yaxis_title='Rating (0-10)',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Clutch moments distribution
    st.subheader("🎯 Clutch Moments Distribution")
    
    clutch_players = df[df['clutch_moments'] > 0].sort_values('clutch_moments', ascending=False)
    
    if not clutch_players.empty:
        fig = px.treemap(clutch_players,
                         path=['team', 'player_name'],
                         values='clutch_moments',
                         color='pressure_composure',
                         color_continuous_scale='RdYlGn',
                         title='Clutch Moments by Player and Team',
                         height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No clutch moments recorded in this match.")
    
    # Pressure performance by position
    st.subheader("📊 Pressure Performance by Position")
    
    position_analysis = df.groupby('position').agg({
        'pressure_performance': 'mean',
        'pressure_composure': 'mean',
        'clutch_moments': 'sum',
        'player_name': 'count'
    }).reset_index()
    
    position_analysis.columns = ['Position', 'Avg Pressure Perf', 'Avg Composure', 
                                  'Total Clutch Moments', 'Player Count']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Pressure Performance by Position', 
                       'Clutch Moments by Position'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    fig.add_trace(
        go.Bar(x=position_analysis['Position'], 
               y=position_analysis['Avg Pressure Perf'],
               marker_color='#3b82f6',
               name='Pressure Performance'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(labels=position_analysis['Position'], 
               values=position_analysis['Total Clutch Moments'],
               name='Clutch Moments'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Team pressure comparison
    st.subheader("⚔️ Team Pressure Comparison")
    
    team_pressure = df.groupby('team').agg({
        'pressure_performance': 'mean',
        'pressure_composure': 'mean',
        'clutch_moments': 'sum',
        'overall_rating': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    metrics = ['pressure_performance', 'pressure_composure', 'overall_rating']
    metric_names = ['Pressure Performance', 'Composure', 'Overall Rating']
    
    for idx, team in enumerate(team_pressure['team']):
        team_data = team_pressure[team_pressure['team'] == team]
        values = [team_data[m].values[0] for m in metrics]
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_names + [metric_names[0]],
            fill='toself',
            name=team,
            line=dict(width=2),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title='Team Average Performance Comparison',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    best_pressure_player = df.nlargest(1, 'pressure_performance').iloc[0]
    most_clutch_player = df.nlargest(1, 'clutch_moments').iloc[0]
    most_composed_player = df.nlargest(1, 'pressure_composure').iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"**🔥 Best Under Pressure**\n\n{best_pressure_player['player_name']}\n\n"
                  f"Score: {best_pressure_player['pressure_performance']:.1f}/10")
    
    with col2:
        st.success(f"**🎯 Most Clutch**\n\n{most_clutch_player['player_name']}\n\n"
                  f"{int(most_clutch_player['clutch_moments'])} moments")
    
    with col3:
        st.success(f"**😌 Most Composed**\n\n{most_composed_player['player_name']}\n\n"
                  f"Score: {most_composed_player['pressure_composure']:.1f}/10")

# Main Dashboard
def main():
    st.title("⚽ Football Analysis Dashboard")
    st.caption("Comprehensive match and player performance analysis")
    
    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Load data
    analysis_data, status_message = load_latest_analysis_json()
    
    if status_message.startswith("✅"):
        st.success(status_message)
    else:
        st.error(status_message)
    
    if analysis_data is None:
        st.error("Cannot load analysis data. Please ensure JSON files exist in the 'outputfoot' directory.")
        st.info("📁 Expected file structure: `outputfoot/*.json`")
        return
    
    # Create player dataframe
    player_df = create_player_dataframe(analysis_data)
    
    if player_df.empty:
        st.warning("No player performance data found in the JSON file.")
        return
    
    # Generate rankings
    player_df = generate_overall_rankings(player_df)
    player_df = generate_pressure_rankings(player_df)
    
    # Create tabs
    tabs = st.tabs([
        "📊 Match Overview",
        "🏆 Overall Leaderboard",
        "🔥 Pressure Leaderboard",
        "👤 Player Reports",
        "🆚 Head-to-Head",
        "🧠 Mental & Pressure"
    ])
    
    with tabs[0]:
        display_match_overview(analysis_data, player_df)
    
    with tabs[1]:
        display_overall_leaderboard(player_df)
    
    with tabs[2]:
        display_pressure_leaderboard(player_df)
    
    with tabs[3]:
        display_player_reports(player_df, analysis_data)
    
    with tabs[4]:
        display_head_to_head(player_df)
    
    with tabs[5]:
        display_mental_pressure_analysis(player_df, analysis_data)

if __name__ == "__main__":
    main()