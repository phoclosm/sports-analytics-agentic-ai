# import streamlit as st
# import pandas as pd
# import json
# import os
# from pathlib import Path
# import glob
# import plotly.express as px
# import plotly.graph_objects as go
# from scipy.stats import rankdata

# OUTPUT_DIR = "outputfoot"

# st.set_page_config(layout="wide", page_title="⚽ Football Analysis Dashboard")

# def load_latest_analysis_json():
#     """Finds and loads the latest two-pass analysis JSON file."""
#     output_path = Path(OUTPUT_DIR)
#     if not output_path.exists():
#         return None, "Output directory not found. Please run football_analyzer.py first."

#     json_files = glob.glob(str(output_path / "*_twopass.json"))
    
#     if not json_files:
#         return None, "No analysis JSON file found. Run football_analyzer.py first."

#     latest_file = max(json_files, key=os.path.getctime)
    
#     try:
#         with open(latest_file, 'r') as f:
#             data = json.load(f)
#         return data, f"Successfully loaded: {Path(latest_file).name}"
#     except Exception as e:
#         return None, f"Error loading file: {e}"

# def create_player_dataframe(merged_data):
#     """Converts player data into DataFrame for visualization."""
#     if not merged_data or 'players' not in merged_data:
#         return pd.DataFrame()
        
#     rows = []
#     for player in merged_data['players']:
#         detection_info = player.get('detection_info', {})
#         detailed_analysis = player.get('detailed_analysis', {})
        
#         row = {
#             'player_id': player.get('player_id', 'Unknown'),
#             'team': player.get('team', 'Unknown'),
#             'position': detection_info.get('position', 'Unknown'),
#             'is_starter': detection_info.get('is_starter', False),
#             'estimated_minutes': detection_info.get('estimated_minutes', 0),
#         }
        
#         # Add segment totals if available
#         totals = detailed_analysis.get('segment_totals', {})
#         for k, v in totals.items():
#             row[f'total_{k}'] = v if v is not None else 0
        
#         # Add pressure metrics from first segment
#         segments = detailed_analysis.get('analyzed_segments', [])
#         if segments and len(segments) > 0:
#             pressure_metrics = segments[0].get('pressure_metrics', {})
#             row['pressure_intensity'] = pressure_metrics.get('pressure_intensity', 0)
#             row['mental_toughness'] = pressure_metrics.get('mental_toughness', 0)
#             row['clutch_factor'] = pressure_metrics.get('clutch_factor', 'N/A')
#         else:
#             row['pressure_intensity'] = 0
#             row['mental_toughness'] = 0
#             row['clutch_factor'] = 'N/A'
            
#         rows.append(row)
        
#     df = pd.DataFrame(rows).fillna(0)
    
#     # Calculate pass accuracy percentage safely
#     if 'total_pass_accuracy_pct' not in df.columns:
#         if 'total_passes_completed' in df.columns and 'total_passes_attempted' in df.columns:
#             df['total_pass_accuracy_pct'] = df.apply(
#                 lambda x: round((x['total_passes_completed'] / x['total_passes_attempted'] * 100), 1) 
#                 if x['total_passes_attempted'] > 0 else 0,
#                 axis=1
#             )
#         else:
#             df['total_pass_accuracy_pct'] = 0
    
#     # Ensure required columns exist
#     required_cols = ['total_goals', 'total_assists', 'total_shots_on_target', 'total_passes_completed']
#     for col in required_cols:
#         if col not in df.columns:
#             df[col] = 0
        
#     return df

# def generate_gamification_rankings(df):
#     """Generates composite Clutch Rank based on performance indicators."""
#     if df.empty or len(df) == 0:
#         return df

#     stats_to_rank = {
#         'goals': 'Goals',
#         'assists': 'Assists',
#         'shots_on_target': 'Shots',
#         'pass_accuracy_pct': 'Pass%',
#         'mental_toughness': 'Toughness'
#     }
    
#     rank_df = pd.DataFrame(index=df.index)

#     for stat, label in stats_to_rank.items():
#         col_name = f'total_{stat}' if stat not in ['mental_toughness', 'pass_accuracy_pct'] else stat
#         if col_name in df.columns and df[col_name].sum() > 0:
#             ranks = rankdata(-df[col_name], method='min') 
#             percentile = (len(df) - ranks + 1) / len(df)
#             rank_df[f'{label} Score'] = (percentile * 100).round(0)
#         else:
#             rank_df[f'{label} Score'] = 50

#     score_cols = [col for col in rank_df.columns if 'Score' in col]
#     if score_cols:
#         rank_df['OVERALL SCORE'] = rank_df[score_cols].mean(axis=1).round(0)
#         final_ranks = rankdata(-rank_df['OVERALL SCORE'], method='min') 
#         rank_df['CLUTCH RANK'] = final_ranks.astype(int)
#     else:
#         rank_df['OVERALL SCORE'] = 50
#         rank_df['CLUTCH RANK'] = 1

#     df = df.join(rank_df)
#     return df.sort_values(by='CLUTCH RANK', ascending=True)

# def display_game_overview(game_info, df):
#     st.header("⚽ Match Overview & Analysis Summary")
    
#     vs = game_info.get('video_structure', {})
#     meta = game_info.get('analysis_metadata', {})
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric("Team A", vs.get('team_a', 'N/A'))
#         st.metric("Total Players Analyzed", len(df))
        
#     with col2:
#         st.metric("Team B", vs.get('team_b', 'N/A'))
#         st.metric("Match Type", vs.get('match_type', 'N/A'))
        
#     with col3:
#         periods = vs.get('periods_present', ['N/A'])
#         st.metric("Periods Present", ', '.join(periods) if isinstance(periods, list) else periods)
#         segments = meta.get('segments_analyzed', [])
#         st.metric("Segments Analyzed", f"{len(segments)} segments")

#     st.subheader("⏱️ Key Moments Timeline")
#     key_moments = game_info.get('key_moments', [])
#     if key_moments and len(key_moments) > 0:
#         moments_df = pd.DataFrame(key_moments)
#         display_cols = [col for col in ['timestamp', 'game_minute', 'period', 'moment_type', 'event_details', 'score_change'] 
#                        if col in moments_df.columns]
#         if display_cols:
#             st.dataframe(moments_df[display_cols], use_container_width=True, hide_index=True)
#     else:
#         st.info("No key moments were flagged in the analysis.")

# def display_player_leaderboard(df):
#     st.header("🏆 Player Leaderboard (Gamified Clutch Rank)")

#     if df.empty:
#         st.warning("No player data available for the leaderboard.")
#         return

#     leaderboard_cols = ['CLUTCH RANK', 'player_id', 'team', 'position', 'OVERALL SCORE', 
#                        'total_goals', 'total_assists', 'total_shots_on_target', 
#                        'total_pass_accuracy_pct', 'mental_toughness', 'pressure_intensity']
    
#     existing_cols = [col for col in leaderboard_cols if col in df.columns]
#     display_df = df[existing_cols].rename(columns={
#         'player_id': 'Player', 
#         'total_goals': 'Goals', 
#         'total_assists': 'Assists', 
#         'total_shots_on_target': 'Shots on Target',
#         'total_pass_accuracy_pct': 'Pass%', 
#         'mental_toughness': 'Toughness (1-10)', 
#         'pressure_intensity': 'Pressure (1-10)', 
#         'team': 'Team',
#         'position': 'Position'
#     })

#     st.dataframe(display_df, use_container_width=True, hide_index=True)
    
#     st.markdown("""
#         <div style='font-size: small; color: grey;'>
#             **CLUTCH RANK** is a composite score based on Goals, Assists, Shots, Pass%, and Mental Toughness.
#         </div>
#     """, unsafe_allow_html=True)

# def display_player_reports(df, merged_data):
#     st.header("👤 Player Reports")
    
#     if df.empty:
#         st.warning("No player data available for detailed reports.")
#         return

#     player_id_list = df['player_id'].unique().tolist()
#     selected_player = st.selectbox("Select a Player", player_id_list)

#     if selected_player:
#         player_row = df[df['player_id'] == selected_player].iloc[0]
        
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.subheader(f"Stats: {selected_player}")
#             st.metric("Team", player_row.get('team', 'N/A'))
#             st.metric("Position", player_row.get('position', 'N/A'))
#             st.metric("Clutch Rank", int(player_row.get('CLUTCH RANK', 0)))
#             st.metric("Goals", f"{player_row.get('total_goals', 0):.0f}")
#             st.metric("Assists", f"{player_row.get('total_assists', 0):.0f}")
#             st.metric("Pass Accuracy", f"{player_row.get('total_pass_accuracy_pct', 0):.1f}%")
#             st.metric("Mental Toughness", f"{player_row.get('mental_toughness', 0):.0f}/10")
            
#         with col2:
#             st.subheader("Performance Radar")
            
#             radar_categories = ['Goals', 'Assists', 'Shots', 'Pass%', 'Toughness']
            
#             r = [
#                 player_row.get('Goals Score', 50),
#                 player_row.get('Assists Score', 50),
#                 player_row.get('Shots Score', 50),
#                 player_row.get('Pass% Score', 50),
#                 player_row.get('Toughness Score', 50)
#             ]
            
#             fig = go.Figure(data=[
#                 go.Scatterpolar(
#                     r=r,
#                     theta=radar_categories,
#                     fill='toself',
#                     name=selected_player
#                 )
#             ])

#             fig.update_layout(
#                 polar=dict(
#                     radialaxis=dict(
#                         visible=True,
#                         range=[0, 100]
#                     )),
#                 showlegend=False
#             )
#             st.plotly_chart(fig, use_container_width=True)

# def display_head_to_head(df):
#     st.header("🆚 Head-to-Head Comparison")
    
#     if df.empty or len(df) < 2:
#         st.info("At least two players are required for Head-to-Head comparison.")
#         return

#     teams = df['team'].unique().tolist()
#     if len(teams) < 2:
#         st.warning("Need players from both teams for comparison.")
#         return

#     team_a = teams[0]
#     team_b = teams[1] if len(teams) > 1 else teams[0]

#     col1, col2 = st.columns(2)
    
#     with col1:
#         player1_list = df[df['team'] == team_a]['player_id'].tolist()
#         if player1_list:
#             player1 = st.selectbox(f"Select Player 1 ({team_a})", player1_list)
#         else:
#             st.warning(f"No players from {team_a}")
#             return
        
#     with col2:
#         player2_list = df[df['team'] == team_b]['player_id'].tolist()
#         if player2_list:
#             player2 = st.selectbox(f"Select Player 2 ({team_b})", player2_list)
#         else:
#             st.warning(f"No players from {team_b}")
#             return

#     if player1 and player2:
#         p1_data = df[df['player_id'] == player1].iloc[0]
#         p2_data = df[df['player_id'] == player2].iloc[0]

#         comp_df = pd.DataFrame({
#             'Metric': ['Goals', 'Assists', 'Shots on Target', 'Pass%', 'Toughness'],
#             player1: [
#                 p1_data.get('total_goals', 0), 
#                 p1_data.get('total_assists', 0), 
#                 p1_data.get('total_shots_on_target', 0), 
#                 p1_data.get('total_pass_accuracy_pct', 0), 
#                 p1_data.get('mental_toughness', 0)
#             ],
#             player2: [
#                 p2_data.get('total_goals', 0), 
#                 p2_data.get('total_assists', 0), 
#                 p2_data.get('total_shots_on_target', 0), 
#                 p2_data.get('total_pass_accuracy_pct', 0), 
#                 p2_data.get('mental_toughness', 0)
#             ],
#         }).set_index('Metric')
        
#         st.subheader(f"{player1} vs {player2}")
#         st.dataframe(comp_df, use_container_width=True)
        
#         plot_df = comp_df.reset_index().melt('Metric', var_name='Player', value_name='Value')
#         fig = px.bar(plot_df, x='Metric', y='Value', color='Player', barmode='group',
#                      title="Key Stat Comparison", height=400)
#         st.plotly_chart(fig, use_container_width=True)

# def display_mental_pressure_analysis(df):
#     st.header("🧠 Mental & Pressure Analysis")
    
#     if df.empty:
#         st.warning("No player data available for pressure analysis.")
#         return
        
#     st.subheader("Pressure vs. Pass Accuracy")
    
#     fig = px.scatter(df, 
#                      x='pressure_intensity', 
#                      y='total_pass_accuracy_pct', 
#                      color='team',
#                      size='total_goals',
#                      hover_name='player_id',
#                      title="Pressure Intensity vs. Pass Accuracy",
#                      labels={'pressure_intensity': 'Pressure Intensity (1-10)', 
#                              'total_pass_accuracy_pct': 'Pass Accuracy %'},
#                      height=500)
#     fig.update_layout(xaxis=dict(range=[0, 10]), yaxis=dict(range=[0, 100]))
#     st.plotly_chart(fig, use_container_width=True)
    
#     st.subheader("Mental Toughness Distribution")
    
#     sorted_df = df.sort_values(by='mental_toughness', ascending=False)
#     fig = px.bar(sorted_df,
#                  x='player_id',
#                  y='mental_toughness',
#                  color='team',
#                  title="Player Mental Toughness Score (1-10)",
#                  labels={'mental_toughness': 'Toughness Score', 'player_id': 'Player'},
#                  height=450)
#     st.plotly_chart(fig, use_container_width=True)
    
#     st.subheader("Goals vs. Pressure Performance")
    
#     fig = px.scatter(df,
#                      x='total_goals',
#                      y='mental_toughness',
#                      color='team',
#                      size='pressure_intensity',
#                      hover_name='player_id',
#                      title="Goal Scoring vs. Mental Toughness",
#                      labels={'total_goals': 'Total Goals', 
#                             'mental_toughness': 'Mental Toughness (1-10)'},
#                      height=450)
#     st.plotly_chart(fig, use_container_width=True)

# # Main Dashboard Logic
# analysis_data, status_message = load_latest_analysis_json()

# st.title("⚽ Two-Pass Football Video Analysis Dashboard")
# st.caption(f"Status: {status_message}")

# if analysis_data is None:
#     st.error("Cannot load analysis data. Please run football_analyzer.py first.")
# else:
#     game_info = analysis_data.get('game_info', {})
    
#     player_df = create_player_dataframe(analysis_data)
    
#     if not player_df.empty:
#         player_df = generate_gamification_rankings(player_df)

#     tab_overview, tab_leaderboard, tab_reports, tab_h2h, tab_pressure = st.tabs([
#         "Match Overview", 
#         "Player Leaderboard", 
#         "Player Reports", 
#         "Head-to-Head Comparison", 
#         "Mental & Pressure Analysis"
#     ])

#     with tab_overview:
#         display_game_overview(game_info, player_df)

#     with tab_leaderboard:
#         display_player_leaderboard(player_df)
        
#     with tab_reports:
#         display_player_reports(player_df, analysis_data)

#     with tab_h2h:
#         display_head_to_head(player_df)

#     with tab_pressure:
#         display_mental_pressure_analysis(player_df)

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import glob
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import rankdata

OUTPUT_DIR = "outputfoot"

st.set_page_config(layout="wide", page_title="⚽ Football Analysis Dashboard")

def load_latest_analysis_json():
    """Finds and loads the latest JSON file from outputfoot directory."""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return None, "Output directory 'outputfoot' not found. Please create the directory and add JSON files."

    # Look for all JSON files in the directory
    json_files = glob.glob(str(output_path / "*.json"))
    
    if not json_files:
        return None, "No JSON files found in 'outputfoot' directory. Please add analysis JSON files."

    # Get the most recently modified file
    latest_file = max(json_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, f"✅ Successfully loaded: {Path(latest_file).name} (Modified: {pd.Timestamp.fromtimestamp(os.path.getmtime(latest_file)).strftime('%Y-%m-%d %H:%M:%S')})"
    except Exception as e:
        return None, f"❌ Error loading file: {e}"

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
            'player_name': player.get('player_name', 'Unknown'),
            'team': player.get('team', 'Unknown'),
            'position': detection_info.get('position', 'Unknown'),
            'is_starter': detection_info.get('is_starter', False),
            'estimated_minutes': detection_info.get('estimated_minutes', 0),
        }
        
        # Add segment totals if available
        totals = detailed_analysis.get('segment_totals', {})
        if totals:
            row['total_goals'] = totals.get('total_goals', 0)
            row['total_assists'] = totals.get('total_assists', 0)
            row['total_shots'] = totals.get('total_shots', 0)
            row['total_shots_on_target'] = totals.get('shots_on_target', 0)
            row['total_passes_completed'] = totals.get('total_passes_completed', 0)
            row['total_passes_attempted'] = totals.get('total_passes_attempted', 0)
            row['total_pass_accuracy_pct'] = totals.get('pass_accuracy_pct', 0)
            row['total_dribbles_successful'] = totals.get('total_dribbles_successful', 0)
            row['total_tackles_won'] = totals.get('total_tackles_won', 0)
            row['total_touches'] = totals.get('total_touches', 0)
            row['total_distance_meters'] = totals.get('total_distance_meters', 0)
            row['fouls_committed'] = totals.get('fouls_committed', 0)
            row['fouls_won'] = totals.get('fouls_won', 0)
            row['yellow_cards'] = totals.get('yellow_cards', 0)
            row['overall_performance_rating'] = totals.get('overall_performance_rating', 0)
        else:
            # Set defaults if no totals available
            row['total_goals'] = 0
            row['total_assists'] = 0
            row['total_shots'] = 0
            row['total_shots_on_target'] = 0
            row['total_passes_completed'] = 0
            row['total_passes_attempted'] = 0
            row['total_pass_accuracy_pct'] = 0
            row['total_dribbles_successful'] = 0
            row['total_tackles_won'] = 0
            row['total_touches'] = 0
            row['total_distance_meters'] = 0
            row['fouls_committed'] = 0
            row['fouls_won'] = 0
            row['yellow_cards'] = 0
            row['overall_performance_rating'] = 0
        
        # Add pressure metrics from first segment
        segments = detailed_analysis.get('analyzed_segments', [])
        if segments and len(segments) > 0:
            pressure_metrics = segments[0].get('pressure_metrics', {})
            row['pressure_intensity'] = pressure_metrics.get('pressure_intensity_rating', 0)
            row['mental_toughness'] = pressure_metrics.get('mental_toughness_rating', 0)
            row['composure_under_pressure'] = pressure_metrics.get('composure_under_pressure', 0)
            row['clutch_factor'] = pressure_metrics.get('clutch_factor', 'N/A')
        else:
            row['pressure_intensity'] = 0
            row['mental_toughness'] = 0
            row['composure_under_pressure'] = 0
            row['clutch_factor'] = 'N/A'
            
        rows.append(row)
        
    df = pd.DataFrame(rows).fillna(0)
    
    # Calculate pass accuracy percentage safely if not already present
    if 'total_pass_accuracy_pct' not in df.columns or df['total_pass_accuracy_pct'].sum() == 0:
        if 'total_passes_completed' in df.columns and 'total_passes_attempted' in df.columns:
            df['total_pass_accuracy_pct'] = df.apply(
                lambda x: round((x['total_passes_completed'] / x['total_passes_attempted'] * 100), 1) 
                if x['total_passes_attempted'] > 0 else 0,
                axis=1
            )
        else:
            df['total_pass_accuracy_pct'] = 0
        
    return df

def generate_gamification_rankings(df):
    """Generates composite Clutch Rank based on performance indicators."""
    if df.empty or len(df) == 0:
        return df

    stats_to_rank = {
        'total_goals': 'Goals',
        'total_assists': 'Assists',
        'total_shots_on_target': 'Shots',
        'total_pass_accuracy_pct': 'Pass%',
        'mental_toughness': 'Toughness'
    }
    
    rank_df = pd.DataFrame(index=df.index)

    for stat, label in stats_to_rank.items():
        if stat in df.columns and df[stat].sum() > 0:
            # Higher is better, so we negate for ranking
            ranks = rankdata(-df[stat].fillna(0), method='min') 
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
    st.header("⚽ Match Overview & Analysis Summary")
    
    vs = game_info.get('video_structure', {})
    meta = game_info.get('analysis_metadata', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Team A", vs.get('team_a', 'N/A'))
        final_score = vs.get('final_score', {})
        if final_score:
            st.metric("Score", f"{final_score.get('team_a', 0)} - {final_score.get('team_b', 0)}")
        
    with col2:
        st.metric("Team B", vs.get('team_b', 'N/A'))
        st.metric("Match Type", vs.get('match_type', 'N/A'))
        
    with col3:
        st.metric("Total Players Analyzed", len(df))
        periods = vs.get('periods_present', ['N/A'])
        st.metric("Periods", ', '.join(periods) if isinstance(periods, list) else periods)

    # Goals Timeline
    st.subheader("⚽ Goals Timeline")
    goals_scored = game_info.get('goals_scored', [])
    if goals_scored:
        goals_df = pd.DataFrame(goals_scored)
        if not goals_df.empty:
            # Create a simple timeline view
            for idx, goal in goals_df.iterrows():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                with col1:
                    st.markdown(f"**{goal.get('game_minute', 'N/A')}**")
                with col2:
                    st.markdown(f"⚽ {goal.get('scorer_name', 'Unknown')}")
                with col3:
                    st.markdown(f"*{goal.get('scoring_team', 'Unknown')}*")
                with col4:
                    score_after = goal.get('score_after', {})
                    st.markdown(f"`{score_after.get('team_a', 0)}-{score_after.get('team_b', 0)}`")

    st.subheader("⏱️ Key Moments Timeline")
    key_moments = game_info.get('key_moments', [])
    if key_moments and len(key_moments) > 0:
        moments_df = pd.DataFrame(key_moments)
        
        # Create a styled dataframe view
        for idx, moment in moments_df.iterrows():
            moment_type = moment.get('moment_type', 'event')
            emoji_map = {
                'goal': '⚽',
                'penalty_goal': '🎯',
                'free_kick_goal': '🎯',
                'injury': '🤕',
                'foul': '⚠️',
                'high_pressure': '🔥'
            }
            emoji = emoji_map.get(moment_type, '📌')
            
            st.markdown(f"{emoji} **{moment.get('game_minute', 'N/A')}** - {moment.get('reason', 'No description')}")
    else:
        st.info("No key moments were flagged in the analysis.")

def display_player_leaderboard(df):
    st.header("🏆 Player Leaderboard (Gamified Clutch Rank)")

    if df.empty:
        st.warning("No player data available for the leaderboard.")
        return

    leaderboard_cols = ['CLUTCH RANK', 'player_name', 'team', 'position', 'OVERALL SCORE', 
                       'total_goals', 'total_assists', 'total_shots_on_target', 
                       'total_pass_accuracy_pct', 'mental_toughness', 'pressure_intensity']
    
    existing_cols = [col for col in leaderboard_cols if col in df.columns]
    display_df = df[existing_cols].copy()
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'player_name': 'Player', 
        'total_goals': 'Goals', 
        'total_assists': 'Assists', 
        'total_shots_on_target': 'Shots on Target',
        'total_pass_accuracy_pct': 'Pass%', 
        'mental_toughness': 'Toughness (1-10)', 
        'pressure_intensity': 'Pressure (1-10)', 
        'team': 'Team',
        'position': 'Position'
    })

    # Add trophy emojis for top 3
    def add_trophy(row):
        if row['CLUTCH RANK'] == 1:
            return '🥇 1'
        elif row['CLUTCH RANK'] == 2:
            return '🥈 2'
        elif row['CLUTCH RANK'] == 3:
            return '🥉 3'
        else:
            return str(int(row['CLUTCH RANK']))
    
    display_df['CLUTCH RANK'] = display_df.apply(add_trophy, axis=1)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
        <div style='font-size: small; color: grey; margin-top: 10px;'>
            <b>CLUTCH RANK</b> is a composite score based on Goals, Assists, Shots, Pass%, and Mental Toughness.
        </div>
    """, unsafe_allow_html=True)

def display_player_reports(df, merged_data):
    st.header("👤 Player Reports")
    
    if df.empty:
        st.warning("No player data available for detailed reports.")
        return

    player_list = df[['player_name', 'team']].apply(lambda x: f"{x['player_name']} ({x['team']})", axis=1).tolist()
    player_names = df['player_name'].tolist()
    
    selected_display = st.selectbox("Select a Player", player_list)
    
    if selected_display:
        # Extract player name from selection
        selected_player_name = selected_display.split(' (')[0]
        player_row = df[df['player_name'] == selected_player_name].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Stats: {player_row['player_name']}")
            st.metric("Team", player_row.get('team', 'N/A'))
            st.metric("Position", player_row.get('position', 'N/A'))
            st.metric("Clutch Rank", f"#{int(player_row.get('CLUTCH RANK', 0))}")
            st.metric("Overall Score", f"{player_row.get('OVERALL SCORE', 0):.0f}/100")
            
            st.markdown("---")
            
            st.metric("Goals", f"{player_row.get('total_goals', 0):.0f}")
            st.metric("Assists", f"{player_row.get('total_assists', 0):.0f}")
            st.metric("Shots on Target", f"{player_row.get('total_shots_on_target', 0):.0f}")
            st.metric("Pass Accuracy", f"{player_row.get('total_pass_accuracy_pct', 0):.1f}%")
            st.metric("Mental Toughness", f"{player_row.get('mental_toughness', 0):.0f}/10")
            st.metric("Pressure Intensity", f"{player_row.get('pressure_intensity', 0):.0f}/10")
            
        with col2:
            st.subheader("Performance Radar")
            
            radar_categories = ['Goals', 'Assists', 'Shots', 'Pass%', 'Toughness']
            
            r = [
                player_row.get('Goals Score', 50),
                player_row.get('Assists Score', 50),
                player_row.get('Shots Score', 50),
                player_row.get('Pass% Score', 50),
                player_row.get('Toughness Score', 50)
            ]
            
            fig = go.Figure(data=[
                go.Scatterpolar(
                    r=r,
                    theta=radar_categories,
                    fill='toself',
                    name=player_row['player_name'],
                    line=dict(color='#3b82f6', width=2),
                    fillcolor='rgba(59, 130, 246, 0.3)'
                )
            ])

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional stats
            st.subheader("Additional Statistics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Touches", f"{player_row.get('total_touches', 0):.0f}")
                st.metric("Distance (m)", f"{player_row.get('total_distance_meters', 0):.0f}")
                st.metric("Dribbles", f"{player_row.get('total_dribbles_successful', 0):.0f}")
            with col_b:
                st.metric("Tackles Won", f"{player_row.get('total_tackles_won', 0):.0f}")
                st.metric("Fouls", f"{player_row.get('fouls_committed', 0):.0f}")
                st.metric("Yellow Cards", f"{player_row.get('yellow_cards', 0):.0f}")

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
        player1_list = df[df['team'] == team_a]['player_name'].tolist()
        if player1_list:
            player1_name = st.selectbox(f"Select Player 1 ({team_a})", player1_list, key='p1')
        else:
            st.warning(f"No players from {team_a}")
            return
        
    with col2:
        player2_list = df[df['team'] == team_b]['player_name'].tolist()
        if player2_list:
            player2_name = st.selectbox(f"Select Player 2 ({team_b})", player2_list, key='p2')
        else:
            st.warning(f"No players from {team_b}")
            return

    if player1_name and player2_name:
        p1_data = df[df['player_name'] == player1_name].iloc[0]
        p2_data = df[df['player_name'] == player2_name].iloc[0]

        st.subheader(f"{player1_name} vs {player2_name}")
        
        comp_df = pd.DataFrame({
            'Metric': ['Goals', 'Assists', 'Shots on Target', 'Pass%', 'Toughness', 'Overall Score'],
            player1_name: [
                p1_data.get('total_goals', 0), 
                p1_data.get('total_assists', 0), 
                p1_data.get('total_shots_on_target', 0), 
                p1_data.get('total_pass_accuracy_pct', 0), 
                p1_data.get('mental_toughness', 0),
                p1_data.get('OVERALL SCORE', 0)
            ],
            player2_name: [
                p2_data.get('total_goals', 0), 
                p2_data.get('total_assists', 0), 
                p2_data.get('total_shots_on_target', 0), 
                p2_data.get('total_pass_accuracy_pct', 0), 
                p2_data.get('mental_toughness', 0),
                p2_data.get('OVERALL SCORE', 0)
            ],
        }).set_index('Metric')
        
        st.dataframe(comp_df, use_container_width=True)
        
        # Bar chart comparison
        plot_df = comp_df.reset_index().melt('Metric', var_name='Player', value_name='Value')
        fig = px.bar(plot_df, x='Metric', y='Value', color='Player', barmode='group',
                     title="Key Stat Comparison", height=400,
                     color_discrete_map={player1_name: '#3b82f6', player2_name: '#7c3aed'})
        st.plotly_chart(fig, use_container_width=True)

def display_mental_pressure_analysis(df):
    st.header("🧠 Mental & Pressure Analysis")
    
    if df.empty:
        st.warning("No player data available for pressure analysis.")
        return
        
    st.subheader("Pressure Intensity vs. Pass Accuracy")
    
    fig = px.scatter(df, 
                     x='pressure_intensity', 
                     y='total_pass_accuracy_pct', 
                     color='team',
                     size='total_goals',
                     hover_name='player_name',
                     hover_data=['total_goals', 'total_assists', 'mental_toughness'],
                     title="Pressure Intensity vs. Pass Accuracy",
                     labels={'pressure_intensity': 'Pressure Intensity (1-10)', 
                             'total_pass_accuracy_pct': 'Pass Accuracy %'},
                     height=500)
    fig.update_layout(xaxis=dict(range=[0, 11]), yaxis=dict(range=[0, 105]))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Mental Toughness Distribution")
    
    sorted_df = df.sort_values(by='mental_toughness', ascending=False)
    fig = px.bar(sorted_df,
                 x='player_name',
                 y='mental_toughness',
                 color='team',
                 title="Player Mental Toughness Score (1-10)",
                 labels={'mental_toughness': 'Toughness Score', 'player_name': 'Player'},
                 height=450)
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Goals vs. Mental Toughness")
    
    fig = px.scatter(df,
                     x='total_goals',
                     y='mental_toughness',
                     color='team',
                     size='pressure_intensity',
                     hover_name='player_name',
                     hover_data=['total_assists', 'total_pass_accuracy_pct'],
                     title="Goal Scoring vs. Mental Toughness",
                     labels={'total_goals': 'Total Goals', 
                            'mental_toughness': 'Mental Toughness (1-10)'},
                     height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.info("""
    **Key Insights:**
    - Players with higher pressure intensity ratings tend to be involved in critical moments
    - Mental toughness correlates with performance in decisive situations
    - Top scorers demonstrate exceptional composure under pressure
    """)

# Main Dashboard Logic
st.title("⚽ Football Video Analysis Dashboard")
st.caption("Automatically loads the latest JSON file from 'outputfoot' directory")

# Add refresh button
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🔄 Refresh Data"):
        st.rerun()

analysis_data, status_message = load_latest_analysis_json()

if status_message.startswith("✅"):
    st.success(status_message)
else:
    st.error(status_message)

if analysis_data is None:
    st.error("Cannot load analysis data. Please ensure JSON files exist in the 'outputfoot' directory.")
    st.info("Expected file structure: `outputfoot/*.json`")
else:
    game_info = analysis_data.get('game_info', {})
    
    player_df = create_player_dataframe(analysis_data)
    
    if not player_df.empty:
        player_df = generate_gamification_rankings(player_df)

    tab_overview, tab_leaderboard, tab_reports, tab_h2h, tab_pressure = st.tabs([
        "📊 Match Overview", 
        "🏆 Player Leaderboard", 
        "👤 Player Reports", 
        "🆚 Head-to-Head", 
        "🧠 Mental & Pressure"
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

