import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import glob
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import rankdata

OUTPUT_DIR = "output"

st.set_page_config(layout="wide", page_title="🏀 Sports Analysis Dashboard")

def load_latest_analysis_json():
    """Finds and loads the latest two-pass analysis JSON file."""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return None, "Output directory not found. Please run main.py first."

    json_files = glob.glob(str(output_path / "*_twopass.json"))
    
    if not json_files:
        class_files = glob.glob(str(output_path / "*.json"))
        if class_files:
            latest_file = max(class_files, key=os.path.getctime)
            with open(latest_file, 'r') as f:
                data = json.load(f)
            if 'identified_sport' in data:
                return data, f"Classification-only result for {data['identified_sport'].upper()} loaded."
        return None, "No analysis JSON file found. Run main.py first."

    latest_file = max(json_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return data, f"Successfully loaded: {Path(latest_file).name}"
    except Exception as e:
        return None, f"Error loading file: {e}"

def create_player_dataframe(merged_data):
    """Converts player data into DataFrame for visualization."""
    if not merged_data or 'players' not in merged_data:
        return pd.DataFrame()
        
    rows = []
    for player in merged_data['players']:
        detection_info = player.get('detection_info', {})
        detailed_analysis = player.get('detailed_analysis', {})
        
        row = {
            'player_id': player.get('player_id', 'Unknown'),
            'team': player.get('team', 'Unknown'),
            'is_starter': detection_info.get('is_starter', False),
            'estimated_minutes': detection_info.get('estimated_minutes', 0),
        }
        
        # Add segment totals if available
        totals = detailed_analysis.get('segment_totals', {})
        for k, v in totals.items():
            row[f'total_{k}'] = v if v is not None else 0
        
        # Add pressure metrics from first segment
        segments = detailed_analysis.get('analyzed_segments', [])
        if segments and len(segments) > 0:
            pressure_metrics = segments[0].get('pressure_metrics', {})
            row['pressure_intensity'] = pressure_metrics.get('pressure_intensity', 0)
            row['mental_toughness'] = pressure_metrics.get('mental_toughness', 0)
            row['clutch_factor'] = pressure_metrics.get('clutch_factor', 'N/A')
        else:
            row['pressure_intensity'] = 0
            row['mental_toughness'] = 0
            row['clutch_factor'] = 'N/A'
            
        rows.append(row)
        
    df = pd.DataFrame(rows).fillna(0)
    
    # Calculate shooting percentage safely
    if 'total_fga' in df.columns and 'total_fgm' in df.columns:
        df['total_fg_pct'] = df.apply(
            lambda x: round((x['total_fgm'] / x['total_fga'] * 100), 1) if x['total_fga'] > 0 else 0,
            axis=1
        )
    else:
        df['total_fg_pct'] = 0
        
    # Ensure required columns exist
    required_cols = ['total_pts', 'total_ast', 'total_reb']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
        
    return df

def generate_gamification_rankings(df):
    """Generates composite Clutch Rank based on performance indicators."""
    if df.empty or len(df) == 0:
        return df

    stats_to_rank = {
        'pts': 'PTS',
        'reb': 'REB',
        'ast': 'AST',
        'fg_pct': 'FG%',
        'mental_toughness': 'Toughness'
    }
    
    rank_df = pd.DataFrame(index=df.index)

    for stat, label in stats_to_rank.items():
        col_name = f'total_{stat}' if stat not in ['mental_toughness', 'fg_pct'] else stat
        if col_name in df.columns and df[col_name].sum() > 0:
            ranks = rankdata(-df[col_name], method='min') 
            percentile = (len(df) - ranks + 1) / len(df)
            rank_df[f'{label} Score'] = (percentile * 100).round(0)
        else:
            rank_df[f'{label} Score'] = 50

    score_cols = [col for col in rank_df.columns if 'Score' in col]
    if score_cols:
        rank_df['OVERALL SCORE'] = rank_df[score_cols].mean(axis=1).round(0)
        final_ranks = rankdata(-rank_df['OVERALL SCORE'], method='min') 
        rank_df['CLUTCH RANK'] = final_ranks.astype(int)
    else:
        rank_df['OVERALL SCORE'] = 50
        rank_df['CLUTCH RANK'] = 1

    df = df.join(rank_df)
    return df.sort_values(by='CLUTCH RANK', ascending=True)

def display_game_overview(game_info, df):
    st.header("🏟️ Game Overview & Analysis Summary")
    
    vs = game_info.get('video_structure', {})
    meta = game_info.get('analysis_metadata', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Team A", vs.get('team_a', 'N/A'))
        st.metric("Total Players Analyzed", len(df))
        
    with col2:
        st.metric("Team B", vs.get('team_b', 'N/A'))
        st.metric("Analysis Duration", meta.get('total_analysis_duration', 'N/A'))
        
    with col3:
        quarters = vs.get('quarters_present', ['N/A'])
        st.metric("Quarters Present", ', '.join(quarters) if isinstance(quarters, list) else quarters)
        segments = meta.get('segments_analyzed', [])
        st.metric("Segments Analyzed", f"{len(segments)} segments")

    st.subheader("⏱️ Key Moments Timeline")
    key_moments = game_info.get('key_moments', [])
    if key_moments and len(key_moments) > 0:
        moments_df = pd.DataFrame(key_moments)
        display_cols = [col for col in ['timestamp', 'quarter', 'moment_type', 'reason', 'score_differential'] 
                       if col in moments_df.columns]
        if display_cols:
            st.dataframe(moments_df[display_cols].set_index('timestamp'), use_container_width=True)
    else:
        st.info("No key moments were flagged in the analysis.")

def display_player_leaderboard(df):
    st.header("🏆 Player Leaderboard (Gamified Clutch Rank)")

    if df.empty:
        st.warning("No player data available for the leaderboard.")
        return

    leaderboard_cols = ['CLUTCH RANK', 'player_id', 'team', 'OVERALL SCORE', 
                       'total_pts', 'total_ast', 'total_reb', 'total_fg_pct', 
                       'mental_toughness', 'pressure_intensity']
    
    existing_cols = [col for col in leaderboard_cols if col in df.columns]
    display_df = df[existing_cols].rename(columns={
        'player_id': 'Player', 'total_pts': 'PTS', 'total_ast': 'AST', 
        'total_reb': 'REB', 'total_fg_pct': 'FG%', 
        'mental_toughness': 'Toughness (1-10)', 
        'pressure_intensity': 'Pressure (1-10)', 'team': 'Team'
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
        <div style='font-size: small; color: grey;'>
            **CLUTCH RANK** is a composite score based on PTS, REB, AST, FG%, and Mental Toughness.
        </div>
    """, unsafe_allow_html=True)

def display_player_reports(df, merged_data):
    st.header("👤 Player Reports")
    
    if df.empty:
        st.warning("No player data available for detailed reports.")
        return

    player_id_list = df['player_id'].unique().tolist()
    selected_player = st.selectbox("Select a Player", player_id_list)

    if selected_player:
        player_row = df[df['player_id'] == selected_player].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Stats: {selected_player}")
            st.metric("Team", player_row.get('team', 'N/A'))
            st.metric("Clutch Rank", int(player_row.get('CLUTCH RANK', 0)))
            st.metric("Total Points", f"{player_row.get('total_pts', 0):.0f}")
            st.metric("Field Goal %", f"{player_row.get('total_fg_pct', 0):.1f}%")
            st.metric("Assists & Rebounds", 
                     f"{player_row.get('total_ast', 0):.0f} AST, {player_row.get('total_reb', 0):.0f} REB")
            st.metric("Mental Toughness", f"{player_row.get('mental_toughness', 0):.0f}/10")
            
        with col2:
            st.subheader("Performance Radar")
            
            radar_categories = ['PTS', 'REB', 'AST', 'FG%', 'Toughness']
            
            r = [
                player_row.get('PTS Score', 50),
                player_row.get('REB Score', 50),
                player_row.get('AST Score', 50),
                player_row.get('FG% Score', 50),
                player_row.get('Toughness Score', 50)
            ]
            
            fig = go.Figure(data=[
                go.Scatterpolar(
                    r=r,
                    theta=radar_categories,
                    fill='toself',
                    name=selected_player
                )
            ])

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

def display_head_to_head(df):
    st.header("🆚 Head-to-Head Comparison")
    
    if df.empty or len(df) < 2:
        st.info("At least two players are required for Head-to-Head comparison.")
        return

    teams = df['team'].unique().tolist()
    if len(teams) < 2:
        st.warning("Need players from both teams for comparison.")
        return

    team_a = teams[0]
    team_b = teams[1] if len(teams) > 1 else teams[0]

    col1, col2 = st.columns(2)
    
    with col1:
        player1_list = df[df['team'] == team_a]['player_id'].tolist()
        if player1_list:
            player1 = st.selectbox(f"Select Player 1 ({team_a})", player1_list)
        else:
            st.warning(f"No players from {team_a}")
            return
        
    with col2:
        player2_list = df[df['team'] == team_b]['player_id'].tolist()
        if player2_list:
            player2 = st.selectbox(f"Select Player 2 ({team_b})", player2_list)
        else:
            st.warning(f"No players from {team_b}")
            return

    if player1 and player2:
        p1_data = df[df['player_id'] == player1].iloc[0]
        p2_data = df[df['player_id'] == player2].iloc[0]

        comp_df = pd.DataFrame({
            'Metric': ['PTS', 'AST', 'REB', 'FG%', 'Toughness'],
            player1: [
                p1_data.get('total_pts', 0), 
                p1_data.get('total_ast', 0), 
                p1_data.get('total_reb', 0), 
                p1_data.get('total_fg_pct', 0), 
                p1_data.get('mental_toughness', 0)
            ],
            player2: [
                p2_data.get('total_pts', 0), 
                p2_data.get('total_ast', 0), 
                p2_data.get('total_reb', 0), 
                p2_data.get('total_fg_pct', 0), 
                p2_data.get('mental_toughness', 0)
            ],
        }).set_index('Metric')
        
        st.subheader(f"{player1} vs {player2}")
        st.dataframe(comp_df, use_container_width=True)
        
        plot_df = comp_df.reset_index().melt('Metric', var_name='Player', value_name='Value')
        fig = px.bar(plot_df, x='Metric', y='Value', color='Player', barmode='group',
                     title="Key Stat Comparison", height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_mental_pressure_analysis(df):
    st.header("🧠 Mental & Pressure Analysis")
    
    if df.empty:
        st.warning("No player data available for pressure analysis.")
        return
        
    st.subheader("Pressure vs. Efficiency")
    
    fig = px.scatter(df, 
                     x='pressure_intensity', 
                     y='total_fg_pct', 
                     color='team',
                     size='total_pts',
                     hover_name='player_id',
                     title="Pressure Intensity vs. Field Goal Percentage",
                     labels={'pressure_intensity': 'Pressure Intensity (1-10)', 
                             'total_fg_pct': 'Field Goal %'},
                     height=500)
    fig.update_layout(xaxis=dict(range=[0, 10]), yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Mental Toughness Distribution")
    
    sorted_df = df.sort_values(by='mental_toughness', ascending=False)
    fig = px.bar(sorted_df,
                 x='player_id',
                 y='mental_toughness',
                 color='team',
                 title="Player Mental Toughness Score (1-10)",
                 labels={'mental_toughness': 'Toughness Score', 'player_id': 'Player'},
                 height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Points vs. Pressure Performance")
    
    fig = px.scatter(df,
                     x='total_pts',
                     y='mental_toughness',
                     color='team',
                     size='pressure_intensity',
                     hover_name='player_id',
                     title="Scoring Output vs. Mental Toughness",
                     labels={'total_pts': 'Total Points', 
                            'mental_toughness': 'Mental Toughness (1-10)'},
                     height=450)
    st.plotly_chart(fig, use_container_width=True)

# Main Dashboard Logic
analysis_data, status_message = load_latest_analysis_json()

st.title("🏀 Two-Pass Basketball Video Analysis Dashboard")
st.caption(f"Status: {status_message}")

if analysis_data is None:
    st.error("Cannot load analysis data. Please run main.py first.")
elif 'identified_sport' in analysis_data and 'game_info' not in analysis_data:
    st.info(f"Classification-only result loaded. Identified sport: **{analysis_data['identified_sport'].upper()}**.")
    st.json(analysis_data)
else:
    game_info = analysis_data.get('game_info', {})
    
    player_df = create_player_dataframe(analysis_data)
    
    if not player_df.empty:
        player_df = generate_gamification_rankings(player_df)

    tab_overview, tab_leaderboard, tab_reports, tab_h2h, tab_pressure = st.tabs([
        "Game Overview", 
        "Player Leaderboard", 
        "Player Reports", 
        "Head-to-Head Comparison", 
        "Mental & Pressure Analysis"
    ])

    with tab_overview:
        display_game_overview(game_info, player_df)

    with tab_leaderboard:
        display_player_leaderboard(player_df)
        
    with tab_reports:
        display_player_reports(player_df, analysis_data)

    with tab_h2h:
        display_head_to_head(player_df)

    with tab_pressure:
        display_mental_pressure_analysis(player_df)