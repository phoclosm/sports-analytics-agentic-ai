import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import glob
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import rankdata
from datetime import datetime
import numpy as np

OUTPUT_DIR = "output"
AGE_DATA_FILE = "nba_age_data.csv"
HISTORICAL_DATA_DIR = "C:/wierdapproach/historicaldata/basketball"  # Directory containing basketball_data_YYYY.json files

st.set_page_config(layout="wide", page_title="🏀 Sports Analysis Dashboard")

@st.cache_data
def load_age_data():
    """Load NBA player age data from CSV."""
    if not os.path.exists(AGE_DATA_FILE):
        return None
    try:
        age_df = pd.read_csv(AGE_DATA_FILE)
        age_df.columns = age_df.columns.str.strip()
        if 'Player' in age_df.columns and 'Birth Date' in age_df.columns:
            age_df['Birth Date'] = pd.to_datetime(age_df['Birth Date'], errors='coerce')
            return age_df
        else:
            st.warning(f"Expected columns 'Player' and 'Birth Date' in {AGE_DATA_FILE}")
            return None
    except Exception as e:
        st.error(f"Error loading age data: {e}")
        return None

@st.cache_data
def load_historical_data():
    """Load all historical basketball data from 2008-2015."""
    historical_files = glob.glob(os.path.join(HISTORICAL_DATA_DIR, "basketball_data_*.json"))
    historical_data = []
    
    for file_path in sorted(historical_files):
        try:
            year = int(Path(file_path).stem.split('_')[-1])
            if 2008 <= year <= 2015:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['source_year'] = year
                    historical_data.append(data)
        except Exception as e:
            st.warning(f"Could not load {file_path}: {e}")
    
    return historical_data

def extract_player_name(player_id):
    """Extract clean player name from player_id (removes jersey number)."""
    import re
    cleaned = re.sub(r'#\d+\s*', '', player_id).strip()
    return cleaned

def build_historical_player_dataframe(historical_data):
    """Build a comprehensive historical dataframe for all players across years."""
    all_records = []
    
    for game_data in historical_data:
        year = game_data.get('source_year')
        players = game_data.get('players', [])
        
        for player in players:
            player_id = player.get('player_id', 'Unknown')
            player_name = extract_player_name(player_id)
            team = player.get('team', 'Unknown')
            
            detailed_analysis = player.get('detailed_analysis', {})
            segment_totals = detailed_analysis.get('segment_totals', {})
            
            # Get pressure metrics from first segment if available
            analyzed_segments = detailed_analysis.get('analyzed_segments', [])
            pressure_intensity = 0
            mental_toughness = 0
            
            if analyzed_segments and len(analyzed_segments) > 0:
                pressure_metrics = analyzed_segments[0].get('pressure_metrics', {})
                pressure_intensity = pressure_metrics.get('pressure_intensity', 0)
                mental_toughness = pressure_metrics.get('mental_toughness', 0)
            
            record = {
                'year': year,
                'player_name': player_name,
                'player_id': player_id,
                'team': team,
                'pts': segment_totals.get('pts', 0),
                'ast': segment_totals.get('ast', 0),
                'reb': segment_totals.get('reb', 0),
                'fgm': segment_totals.get('fgm', 0),
                'fga': segment_totals.get('fga', 0),
                'fg_pct': segment_totals.get('fg_pct', 0),
                'pressure_intensity': pressure_intensity,
                'mental_toughness': mental_toughness
            }
            
            all_records.append(record)
    
    if not all_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    return df

def calculate_player_growth(historical_df, player_name):
    """Calculate growth metrics for a specific player."""
    player_data = historical_df[historical_df['player_name'] == player_name].sort_values('year')
    
    if len(player_data) < 2:
        return None
    
    growth_metrics = {
        'years_tracked': player_data['year'].tolist(),
        'pts_trend': player_data['pts'].tolist(),
        'ast_trend': player_data['ast'].tolist(),
        'reb_trend': player_data['reb'].tolist(),
        'fg_pct_trend': player_data['fg_pct'].tolist(),
        'mental_toughness_trend': player_data['mental_toughness'].tolist(),
        'pressure_intensity_trend': player_data['pressure_intensity'].tolist(),
    }
    
    # Calculate growth rates
    growth_metrics['pts_growth'] = ((player_data['pts'].iloc[-1] - player_data['pts'].iloc[0]) / 
                                     (player_data['pts'].iloc[0] + 0.001)) * 100
    growth_metrics['ast_growth'] = ((player_data['ast'].iloc[-1] - player_data['ast'].iloc[0]) / 
                                     (player_data['ast'].iloc[0] + 0.001)) * 100
    growth_metrics['mental_growth'] = ((player_data['mental_toughness'].iloc[-1] - 
                                        player_data['mental_toughness'].iloc[0]) / 
                                       (player_data['mental_toughness'].iloc[0] + 0.001)) * 100
    
    # Peak year
    growth_metrics['peak_year'] = player_data.loc[player_data['pts'].idxmax(), 'year']
    growth_metrics['peak_pts'] = player_data['pts'].max()
    
    return growth_metrics

def calculate_player_age(player_name, birth_date, match_year):
    """Calculate player age at the time of the match."""
    if pd.isna(birth_date) or not match_year:
        return None
    try:
        match_year = int(match_year)
        age = match_year - birth_date.year
        return age
    except:
        return None

def merge_age_data(player_df, age_data, match_year):
    """Merge age data with player performance data."""
    if age_data is None or player_df.empty:
        return player_df
    
    player_df['player_name_clean'] = player_df['player_id'].str.replace(r'#\d+\s*', '', regex=True).str.strip()
    age_data['player_name_clean'] = age_data['Player'].str.strip()
    
    merged_df = player_df.merge(
        age_data[['player_name_clean', 'Birth Date']], 
        on='player_name_clean', 
        how='left'
    )
    
    merged_df['age'] = merged_df.apply(
        lambda row: calculate_player_age(row['player_name_clean'], row['Birth Date'], match_year),
        axis=1
    )
    
    return merged_df

def categorize_age(age):
    """Categorize players by age groups."""
    if pd.isna(age):
        return 'Unknown'
    elif age < 23:
        return 'Young (Under 23)'
    elif 23 <= age < 27:
        return 'Prime Early (23-26)'
    elif 27 <= age < 31:
        return 'Prime Peak (27-30)'
    elif 31 <= age < 35:
        return 'Veteran (31-34)'
    else:
        return 'Senior (35+)'

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
        
        totals = detailed_analysis.get('segment_totals', {})
        for k, v in totals.items():
            row[f'total_{k}'] = v if v is not None else 0
        
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
    
    if 'total_fga' in df.columns and 'total_fgm' in df.columns:
        df['total_fg_pct'] = df.apply(
            lambda x: round((x['total_fgm'] / x['total_fga'] * 100), 1) if x['total_fga'] > 0 else 0,
            axis=1
        )
    else:
        df['total_fg_pct'] = 0
        
    required_cols = ['total_pts', 'total_ast', 'total_reb']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
            
    return df

def generate_gamification_rankings(df, historical_df=None):
    """Generates composite Clutch Rank based on performance indicators, including historical context."""
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

    # Add historical performance multiplier if available
    df['historical_multiplier'] = 1.0
    
    if historical_df is not None and not historical_df.empty:
        df['player_name_clean'] = df['player_id'].str.replace(r'#\d+\s*', '', regex=True).str.strip()
        
        for idx, row in df.iterrows():
            player_name = row['player_name_clean']
            player_history = historical_df[historical_df['player_name'] == player_name]
            
            if not player_history.empty:
                # Calculate career average performance
                career_avg_pts = player_history['pts'].mean()
                career_peak_pts = player_history['pts'].max()
                
                # Boost multiplier for consistent high performers
                if career_avg_pts > 5:
                    df.at[idx, 'historical_multiplier'] = 1.2
                if career_peak_pts > 10:
                    df.at[idx, 'historical_multiplier'] = 1.3

    for stat, label in stats_to_rank.items():
        col_name = f'total_{stat}' if stat not in ['mental_toughness', 'fg_pct'] else stat
        if col_name in df.columns and df[col_name].sum() > 0:
            # Apply historical multiplier
            adjusted_values = df[col_name] * df['historical_multiplier']
            ranks = rankdata(-adjusted_values, method='min') 
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
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Team A", vs.get('team_a', 'N/A'))
        
    with col2:
        st.metric("Team B", vs.get('team_b', 'N/A'))
        
    with col3:
        match_year = vs.get('match_year', 'Unknown')
        st.metric("Match Year", match_year if match_year else 'Unknown')
        
    with col4:
        st.metric("Total Players", len(df))

    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Analysis Duration", meta.get('total_analysis_duration', 'N/A'))
        
    with col6:
        quarters = vs.get('quarters_present', ['N/A'])
        st.metric("Quarters Present", ', '.join(quarters) if isinstance(quarters, list) else quarters)
        
    with col7:
        segments = meta.get('segments_analyzed', [])
        st.metric("Segments Analyzed", f"{len(segments)} segments")

def display_key_moments(game_info):
    st.header("⏱️ Key Moments Timeline")
    key_moments = game_info.get('key_moments', [])
    if key_moments and len(key_moments) > 0:
        moments_df = pd.DataFrame(key_moments)
        display_cols = [col for col in ['timestamp', 'quarter', 'moment_type', 'players_involved', 'event_details'] 
                        if col in moments_df.columns]
        if display_cols:
            st.dataframe(moments_df[display_cols], width='stretch', hide_index=True)
    else:
        st.info("No key moments were flagged in the analysis.")

def display_player_leaderboard(df):
    st.header("🏆 Player Leaderboard (Gamified Clutch Rank)")

    if df.empty:
        st.warning("No player data available for the leaderboard.")
        return

    leaderboard_cols = ['CLUTCH RANK', 'player_id', 'team']
    
    if 'age' in df.columns:
        leaderboard_cols.append('age')
    
    leaderboard_cols.extend(['OVERALL SCORE', 'total_pts', 'total_ast', 'total_reb', 
                             'total_fg_pct', 'mental_toughness', 'pressure_intensity'])
    
    existing_cols = [col for col in leaderboard_cols if col in df.columns]
    
    rename_dict = {
        'player_id': 'Player', 
        'total_pts': 'PTS', 
        'total_ast': 'AST', 
        'total_reb': 'REB', 
        'total_fg_pct': 'FG%', 
        'mental_toughness': 'Toughness (1-10)', 
        'pressure_intensity': 'Pressure (1-10)', 
        'team': 'Team'
    }
    
    if 'age' in existing_cols:
        rename_dict['age'] = 'Age'
    
    display_df = df[existing_cols].rename(columns=rename_dict)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
        <div style='font-size: small; color: grey;'>
            <b>CLUTCH RANK</b> is a composite score based on PTS, REB, AST, FG%, Mental Toughness, and historical performance.
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
            
            if 'age' in player_row and pd.notna(player_row['age']):
                st.metric("Age", f"{int(player_row['age'])} years")
                st.metric("Age Category", player_row.get('age_category', 'N/A'))
            
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

        metrics = ['PTS', 'AST', 'REB', 'FG%', 'Toughness']
        p1_values = [
            p1_data.get('total_pts', 0), 
            p1_data.get('total_ast', 0), 
            p1_data.get('total_reb', 0), 
            p1_data.get('total_fg_pct', 0), 
            p1_data.get('mental_toughness', 0)
        ]
        p2_values = [
            p2_data.get('total_pts', 0), 
            p2_data.get('total_ast', 0), 
            p2_data.get('total_reb', 0), 
            p2_data.get('total_fg_pct', 0), 
            p2_data.get('mental_toughness', 0)
        ]
        
        if 'age' in df.columns:
            metrics.append('Age')
            p1_values.append(p1_data.get('age', 0) if pd.notna(p1_data.get('age')) else 0)
            p2_values.append(p2_data.get('age', 0) if pd.notna(p2_data.get('age')) else 0)
        
        comp_df = pd.DataFrame({
            'Metric': metrics,
            player1: p1_values,
            player2: p2_values,
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

def display_age_analysis(df):
    st.header("📊 Age-Based Performance Analysis")
    
    if df.empty:
        st.warning("No player data available for age analysis.")
        return
    
    if 'age' not in df.columns or df['age'].isna().all():
        st.warning(f"Age data not available. Please ensure '{AGE_DATA_FILE}' exists with 'Player' and 'Birth Date' columns.")
        return
    
    df_with_age = df[df['age'].notna()].copy()
    
    if df_with_age.empty:
        st.warning("No players with valid age data found.")
        return
    
    df_with_age['age_category'] = df_with_age['age'].apply(categorize_age)
    
    st.subheader("🎯 Age Distribution Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_age = df_with_age['age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    with col2:
        youngest = df_with_age['age'].min()
        youngest_player = df_with_age[df_with_age['age'] == youngest]['player_id'].iloc[0]
        st.metric("Youngest Player", f"{youngest_player} ({int(youngest)})")
    
    with col3:
        oldest = df_with_age['age'].max()
        oldest_player = df_with_age[df_with_age['age'] == oldest]['player_id'].iloc[0]
        st.metric("Oldest Player", f"{oldest_player} ({int(oldest)})")
    
    st.subheader("📈 Age Distribution")
    fig = px.histogram(df_with_age, x='age', nbins=15,
                       title="Player Age Distribution",
                       labels={'age': 'Age (years)', 'count': 'Number of Players'},
                       color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🏅 Performance by Age Category")
    
    age_category_order = ['Young (Under 23)', 'Prime Early (23-26)', 
                          'Prime Peak (27-30)', 'Veteran (31-34)', 'Senior (35+)']
    
    present_categories = [cat for cat in age_category_order if cat in df_with_age['age_category'].unique()]
    
    age_stats = df_with_age.groupby('age_category').agg({
        'total_pts': 'mean',
        'total_ast': 'mean',
        'total_reb': 'mean',
        'total_fg_pct': 'mean',
        'mental_toughness': 'mean',
        'OVERALL SCORE': 'mean',
        'player_id': 'count'
    }).round(2).reset_index()
    
    age_stats.columns = ['Age Category', 'Avg PTS', 'Avg AST', 'Avg REB', 
                         'Avg FG%', 'Avg Toughness', 'Avg Overall Score', 'Player Count']
    
    age_stats['category_order'] = age_stats['Age Category'].map({cat: i for i, cat in enumerate(age_category_order)})
    age_stats = age_stats.sort_values('category_order').drop('category_order', axis=1)
    
    st.dataframe(age_stats, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(age_stats, x='Age Category', y='Avg PTS',
                     title="Average Points by Age Category",
                     labels={'Avg PTS': 'Average Points'},
                     color='Avg PTS',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(age_stats, x='Age Category', y='Avg Overall Score',
                     title="Average Overall Score by Age Category",
                     labels={'Avg Overall Score': 'Average Overall Score'},
                     color='Avg Overall Score',
                     color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)

def display_player_growth_analysis(df, historical_df):
    st.header("📈 Player Growth & Performance Trajectory Analysis")
    
    if historical_df.empty:
        st.warning("No historical data available. Please ensure basketball_data_YYYY.json files (2008-2015) are present.")
        return
    
    if df.empty:
        st.warning("No current player data available.")
        return
    
    # Get list of players in current match who have historical data
    df['player_name_clean'] = df['player_id'].str.replace(r'#\d+\s*', '', regex=True).str.strip()
    available_players = []
    
    for player_name in df['player_name_clean'].unique():
        if player_name in historical_df['player_name'].values:
            available_players.append(player_name)
    
    if not available_players:
        st.warning("No players in the current match have historical data available (2008-2015).")
        return
    
    st.subheader("🎯 Select Player for Growth Analysis")
    
    # Map clean names back to full player IDs for display
    player_display_map = dict(zip(df['player_name_clean'], df['player_id']))
    display_names = [player_display_map[name] for name in available_players]
    
    selected_display = st.selectbox("Choose a player to analyze their career trajectory:", display_names)
    selected_player = extract_player_name(selected_display)
    
    if selected_player:
        # Get player's historical data
        player_history = historical_df[historical_df['player_name'] == selected_player].sort_values('year')
        current_player_data = df[df['player_name_clean'] == selected_player].iloc[0]
        
        # Calculate growth metrics
        growth_data = calculate_player_growth(historical_df, selected_player)
        
        if growth_data is None:
            st.warning("Insufficient historical data for this player.")
            return
        
        # Overview metrics
        st.subheader(f"📊 Career Overview: {selected_display}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Years Tracked", f"{len(growth_data['years_tracked'])} seasons")
            st.caption(f"{min(growth_data['years_tracked'])} - {max(growth_data['years_tracked'])}")
        
        with col2:
            st.metric("Points Growth", f"{growth_data['pts_growth']:.1f}%",
                     delta=f"{growth_data['pts_growth']:.1f}%")
        
        with col3:
            st.metric("Peak Year", growth_data['peak_year'])
            st.caption(f"Peak: {growth_data['peak_pts']:.1f} PTS")
        
        with col4:
            st.metric("Mental Growth", f"{growth_data['mental_growth']:.1f}%",
                     delta=f"{growth_data['mental_growth']:.1f}%")
        
        # Performance Trends
        st.subheader("📈 Performance Trends Over Time")
        
        # Points Trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=growth_data['years_tracked'],
            y=growth_data['pts_trend'],
            mode='lines+markers',
            name='Points',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        
        # Add current season point
        current_year = df[df['player_name_clean'] == selected_player]['year'].iloc[0] if 'year' in df.columns else 'Current'
        if current_year != 'Current':
            fig.add_trace(go.Scatter(
                x=[current_year],
                y=[current_player_data['total_pts']],
                mode='markers',
                name='Current Match',
                marker=dict(size=15, color='red', symbol='star')
            ))
        
        fig.update_layout(
            title="Points Per Game Trend",
            xaxis_title="Year",
            yaxis_title="Points",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-metric comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=growth_data['years_tracked'], y=growth_data['ast_trend'],
                                    mode='lines+markers', name='Assists', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=growth_data['years_tracked'], y=growth_data['reb_trend'],
                                    mode='lines+markers', name='Rebounds', line=dict(color='orange')))
            fig.update_layout(title="Assists & Rebounds Trend", xaxis_title="Year", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=growth_data['years_tracked'], y=growth_data['fg_pct_trend'],
                                    mode='lines+markers', name='FG%', line=dict(color='purple')))
            fig.update_layout(title="Field Goal % Trend", xaxis_title="Year", yaxis_title="FG%", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Mental Toughness Evolution
        st.subheader("🧠 Mental Toughness & Pressure Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=growth_data['years_tracked'],
                y=growth_data['mental_toughness_trend'],
                mode='lines+markers',
                name='Mental Toughness',
                line=dict(color='#d62728', width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Mental Toughness Evolution (1-10)",
                xaxis_title="Year",
                yaxis_title="Mental Toughness Score",
                yaxis=dict(range=[0, 10]),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=growth_data['years_tracked'],
                y=growth_data['pressure_intensity_trend'],
                mode='lines+markers',
                name='Pressure Intensity',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Pressure Situations Faced (1-10)",
                xaxis_title="Year",
                yaxis_title="Pressure Intensity",
                yaxis=dict(range=[0, 10]),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Growth vs Decline Analysis
        st.subheader("📊 Growth & Decline Patterns")
        
        # Calculate year-over-year changes
        pts_changes = np.diff(growth_data['pts_trend'])
        years_for_changes = growth_data['years_tracked'][1:]
        
        growth_years = [years_for_changes[i] for i in range(len(pts_changes)) if pts_changes[i] > 0]
        decline_years = [years_for_changes[i] for i in range(len(pts_changes)) if pts_changes[i] < 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Growth Periods**")
            if growth_years:
                for year in growth_years:
                    idx = growth_data['years_tracked'].index(year)
                    change = growth_data['pts_trend'][idx] - growth_data['pts_trend'][idx-1]
                    st.success(f"**{year}**: +{change:.1f} points")
            else:
                st.info("No significant growth periods identified")
        
        with col2:
            st.markdown("**🔴 Decline Periods**")
            if decline_years:
                for year in decline_years:
                    idx = growth_data['years_tracked'].index(year)
                    change = growth_data['pts_trend'][idx] - growth_data['pts_trend'][idx-1]
                    st.warning(f"**{year}**: {change:.1f} points")
            else:
                st.info("No significant decline periods identified")
        
        # Age-wise Performance Analysis
        if 'age' in current_player_data and pd.notna(current_player_data['age']):
            st.subheader("🎂 Age-Based Performance Analysis")
            
            current_age = int(current_player_data['age'])
            age_category = categorize_age(current_age)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Age", f"{current_age} years")
                st.caption(f"Category: {age_category}")
            
            with col2:
                # Compare to historical average at same age
                if len(player_history) > 0:
                    avg_pts_career = player_history['pts'].mean()
                    current_vs_avg = ((current_player_data['total_pts'] / avg_pts_career - 1) * 100) if avg_pts_career > 0 else 0
                    st.metric("vs Career Avg", f"{current_vs_avg:+.1f}%")
                    st.caption(f"Career Avg: {avg_pts_career:.1f} PTS")
            
            with col3:
                peak_age_idx = growth_data['pts_trend'].index(max(growth_data['pts_trend']))
                peak_age_year = growth_data['years_tracked'][peak_age_idx]
                st.metric("Peak Performance", f"Year {peak_age_year}")
                st.caption(f"{max(growth_data['pts_trend']):.1f} PTS")
        
        # Future Projection (Simple trend-based)
        st.subheader("🔮 Performance Projection")
        
        if len(growth_data['pts_trend']) >= 3:
            # Simple linear projection for next 2 years
            recent_trend = growth_data['pts_trend'][-3:]
            trend_slope = (recent_trend[-1] - recent_trend[0]) / 2
            
            projection_years = [max(growth_data['years_tracked']) + 1, max(growth_data['years_tracked']) + 2]
            projected_pts = [growth_data['pts_trend'][-1] + trend_slope, growth_data['pts_trend'][-1] + 2*trend_slope]
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=growth_data['years_tracked'],
                y=growth_data['pts_trend'],
                mode='lines+markers',
                name='Historical Performance',
                line=dict(color='blue', width=2)
            ))
            
            # Projection
            fig.add_trace(go.Scatter(
                x=projection_years,
                y=projected_pts,
                mode='lines+markers',
                name='Projected Performance',
                line=dict(color='lightblue', width=2, dash='dash'),
                marker=dict(size=10, symbol='diamond')
            ))
            
            fig.update_layout(
                title="Points Projection (Next 2 Seasons)",
                xaxis_title="Year",
                yaxis_title="Points",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if trend_slope > 0:
                st.success(f"📈 Positive trajectory: Player showing upward trend (+{trend_slope:.2f} pts/year)")
            elif trend_slope < 0:
                st.warning(f"📉 Declining trajectory: Player showing downward trend ({trend_slope:.2f} pts/year)")
            else:
                st.info("➡️ Stable performance: Player maintaining consistent output")
        
        # Historical Detailed Table
        st.subheader("📋 Historical Performance Details")
        
        history_display = player_history[['year', 'team', 'pts', 'ast', 'reb', 'fg_pct', 'mental_toughness', 'pressure_intensity']].copy()
        history_display.columns = ['Year', 'Team', 'PTS', 'AST', 'REB', 'FG%', 'Mental Toughness', 'Pressure']
        history_display = history_display.sort_values('Year', ascending=False)
        
        st.dataframe(history_display, use_container_width=True, hide_index=True)
    
    # League-wide Growth Comparison
    st.markdown("---")
    st.subheader("🏀 League-wide Growth Comparison")
    
    # Calculate growth rates for all players with historical data
    growth_comparison = []
    
    for player_name in historical_df['player_name'].unique():
        player_data = historical_df[historical_df['player_name'] == player_name].sort_values('year')
        
        if len(player_data) >= 2:
            pts_growth = ((player_data['pts'].iloc[-1] - player_data['pts'].iloc[0]) / 
                         (player_data['pts'].iloc[0] + 0.001)) * 100
            
            mental_growth = ((player_data['mental_toughness'].iloc[-1] - player_data['mental_toughness'].iloc[0]) / 
                           (player_data['mental_toughness'].iloc[0] + 0.001)) * 100
            
            growth_comparison.append({
                'Player': player_name,
                'PTS Growth %': pts_growth,
                'Mental Growth %': mental_growth,
                'Avg PTS': player_data['pts'].mean(),
                'Years Tracked': len(player_data)
            })
    
    if growth_comparison:
        growth_df = pd.DataFrame(growth_comparison).sort_values('PTS Growth %', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Fastest Growing Players (Points)**")
            top_growth = growth_df.head(10)
            fig = px.bar(top_growth, x='Player', y='PTS Growth %',
                        color='PTS Growth %',
                        color_continuous_scale='Greens',
                        title="Highest Points Growth Rate")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("**Top 10 Most Declined Players (Points)**")
            bottom_growth = growth_df.tail(10).sort_values('PTS Growth %')
            fig = px.bar(bottom_growth, x='Player', y='PTS Growth %',
                        color='PTS Growth %',
                        color_continuous_scale='Reds',
                        title="Highest Points Decline Rate")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, width='stretch')
        
        # Mental toughness growth leaders
        st.markdown("**Mental Toughness Development Leaders**")
        mental_leaders = growth_df.sort_values('Mental Growth %', ascending=False).head(10)
        
        fig = px.scatter(mental_leaders,
                        x='PTS Growth %',
                        y='Mental Growth %',
                        size='Avg PTS',
                        text='Player',
                        title="Points Growth vs Mental Toughness Growth",
                        labels={'PTS Growth %': 'Points Growth %', 'Mental Growth %': 'Mental Toughness Growth %'})
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

# Main Application Logic
def main():
    st.title("🏀 BasketBall Comprehensive Performance Analysis")

    # Load all data
    age_data = load_age_data()
    historical_data_list = load_historical_data()
    historical_df = build_historical_player_dataframe(historical_data_list)
    data, message = load_latest_analysis_json()

    st.info(message)

    if data:
        if 'identified_sport' in data and 'players' not in data:
            st.header(f"Sport Identified: {data['identified_sport'].upper()}")
            st.warning("This analysis file only contains sport classification. Run the full two-pass analysis for detailed player stats.")
            st.json(data)
        
        elif 'game_info' in data and 'players' in data:
            game_info = data.get('game_info', {})
            video_structure = game_info.get('video_structure', {})
            match_year = video_structure.get('match_year', None)

            # Create base DataFrame
            player_df = create_player_dataframe(data)
            
            if player_df.empty:
                st.error("Failed to parse player data from the JSON file.")
            else:
                # Add year to current data
                player_df['year'] = match_year
                
                # Merge Age Data
                player_df = merge_age_data(player_df, age_data, match_year)
                
                # Generate Gamification Ranks with historical context
                player_df_ranked = generate_gamification_rankings(player_df, historical_df)

                # Add age category
                if 'age' in player_df_ranked.columns:
                    player_df_ranked['age_category'] = player_df_ranked['age'].apply(categorize_age)

                # Sidebar Navigation
                st.sidebar.title("Navigation")
                page_options = [
                    "Game Overview", 
                    "Player Leaderboard", 
                    "Player Reports", 
                    "Head-to-Head", 
                    "Pressure Analysis",
                    "Age Analysis",
                    "Player Growth & Trajectory",
                    "Key Moments",
                    "Raw Data"
                ]
                page = st.sidebar.radio("Go to", page_options)
                
                st.sidebar.markdown("---")
                st.sidebar.info("This dashboard visualizes current match analysis with historical player performance data (2008-2015).")
                
                if not historical_df.empty:
                    st.sidebar.success(f"✅ Historical data loaded: {len(historical_df['player_name'].unique())} players across {len(historical_df['year'].unique())} seasons")
                else:
                    st.sidebar.warning("⚠️ No historical data found. Add basketball_data_YYYY.json files for enhanced analysis.")

                # Page Routing
                if page == "Game Overview":
                    display_game_overview(game_info, player_df_ranked)

                elif page == "Player Leaderboard":
                    display_player_leaderboard(player_df_ranked)

                elif page == "Player Reports":
                    display_player_reports(player_df_ranked, data)

                elif page == "Head-to-Head":
                    display_head_to_head(player_df_ranked)

                elif page == "Pressure Analysis":
                    display_mental_pressure_analysis(player_df_ranked)
                
                elif page == "Age Analysis":
                    display_age_analysis(player_df_ranked)
                
                elif page == "Player Growth & Trajectory":
                    display_player_growth_analysis(player_df_ranked, historical_df)
                
                elif page == "Key Moments":
                    display_key_moments(game_info)

                elif page == "Raw Data":
                    st.header("Raw JSON Output (Current Match)")
                    st.json(data)
                    
                    st.header("Processed Player DataFrame (Current Match)")
                    st.dataframe(player_df_ranked)
                    
                    if not historical_df.empty:
                        st.header("Historical Player Data (2008-2015)")
                        st.dataframe(historical_df)

        else:
            st.error("The loaded JSON file is in an unrecognized format.")
            st.json(data)
    else:
        st.warning("Could not load any analysis data. Please run the analysis script (main.py) first and ensure output files are in the 'output' directory.")
        st.info(f"Looking for files in: {os.path.abspath(OUTPUT_DIR)}")
        st.info(f"Looking for age data at: {os.path.abspath(AGE_DATA_FILE)}")
        st.info(f"Looking for historical data in: {os.path.abspath(HISTORICAL_DATA_DIR)}/basketball_data_*.json")


if __name__ == "__main__":
    main()