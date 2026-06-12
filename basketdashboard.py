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

OUTPUT_DIR = "output"
AGE_DATA_FILE = "nba_age_data.csv"

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
    
    # Clean player names for matching
    player_df['player_name_clean'] = player_df['player_id'].str.replace(r'#\d+\s*', '', regex=True).str.strip()
    age_data['player_name_clean'] = age_data['Player'].str.strip()
    
    # Merge on cleaned names
    merged_df = player_df.merge(
        age_data[['player_name_clean', 'Birth Date']], 
        on='player_name_clean', 
        how='left'
    )
    
    # Calculate ages
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
            <b>CLUTCH RANK</b> is a composite score based on PTS, REB, AST, FG%, and Mental Toughness.
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
    
    # Check if age data is available
    if 'age' not in df.columns or df['age'].isna().all():
        st.warning(f"Age data not available. Please ensure '{AGE_DATA_FILE}' exists with 'Player' and 'Birth Date' columns.")
        return
    
    # Filter out players without age data
    df_with_age = df[df['age'].notna()].copy()
    
    if df_with_age.empty:
        st.warning("No players with valid age data found.")
        return
    
    # Add age categories
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
    
    # Age distribution chart
    st.subheader("📈 Age Distribution")
    fig = px.histogram(df_with_age, x='age', nbins=15,
                       title="Player Age Distribution",
                       labels={'age': 'Age (years)', 'count': 'Number of Players'},
                       color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance by Age Category
    st.subheader("🏅 Performance by Age Category")
    
    age_category_order = ['Young (Under 23)', 'Prime Early (23-26)', 
                          'Prime Peak (27-30)', 'Veteran (31-34)', 'Senior (35+)']
    
    # Filter to only include categories present in data
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
    
    # Reorder by age category
    age_stats['category_order'] = age_stats['Age Category'].map({cat: i for i, cat in enumerate(age_category_order)})
    age_stats = age_stats.sort_values('category_order').drop('category_order', axis=1)
    
    st.dataframe(age_stats, use_container_width=True, hide_index=True)
    
    # Visual comparison across age categories
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
    
    # Best performers by age category
    st.subheader("⭐ Top Performers by Age Category")
    
    category_tabs = st.tabs(present_categories)
    
    for idx, category in enumerate(present_categories):
        with category_tabs[idx]:
            category_df = df_with_age[df_with_age['age_category'] == category].copy()
            
            if not category_df.empty:
                # Sort by overall score
                category_df = category_df.sort_values('OVERALL SCORE', ascending=False)
                
                # Best performer
                best = category_df.iloc[0]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**🥇 Best Performer**")
                    st.metric("Player", best['player_id'])
                    st.metric("Age", f"{int(best['age'])} years")
                    st.metric("Overall Score", f"{best['OVERALL SCORE']:.0f}")
                
                with col2:
                    st.markdown(f"**📊 Statistics**")
                    st.metric("Points", f"{best['total_pts']:.1f}")
                    st.metric("FG%", f"{best['total_fg_pct']:.1f}%")
                    st.metric("Clutch Rank", f"#{int(best['CLUTCH RANK'])}")
                
                with col3:
                    st.markdown(f"**🧠 Mental Metrics**")
                    st.metric("Toughness", f"{best['mental_toughness']:.1f}/10")
                    st.metric("Pressure Intensity", f"{best['pressure_intensity']:.1f}/10")
                
                # Worst performer
                if len(category_df) > 1:
                    worst = category_df.iloc[-1]
                    st.markdown("---")
                    st.markdown(f"**⚠️ Needs Improvement**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Player:** {worst['player_id']} (Age: {int(worst['age'])})")
                        st.write(f"**Overall Score:** {worst['OVERALL SCORE']:.0f}")
                    with col2:
                        st.write(f"**Points:** {worst['total_pts']:.1f}")
                        st.write(f"**FG%:** {worst['total_fg_pct']:.1f}%")
                
                # Full category leaderboard
                st.markdown("---")
                st.markdown("**Category Leaderboard**")
                display_cols = ['player_id', 'age', 'OVERALL SCORE', 'total_pts', 
                                'total_fg_pct', 'mental_toughness', 'CLUTCH RANK']
                existing = [col for col in display_cols if col in category_df.columns]
                st.dataframe(category_df[existing], use_container_width=True, hide_index=True)
    
    # Promising Young Players (Under 23)
    st.subheader("🌟 Promising Young Players (Under 26)")
    
    young_players = df_with_age[df_with_age['age'] < 26].copy()
    
    if not young_players.empty:
        young_players = young_players.sort_values('OVERALL SCORE', ascending=False)
        
        st.write(f"Found **{len(young_players)}** players under 23 years old")
        
        # Top 3 promising players
        top_young = young_players.head(3)
        
        cols = st.columns(min(3, len(top_young)))
        
        for idx, (_, player) in enumerate(top_young.iterrows()):
            with cols[idx]:
                st.markdown(f"**#{idx+1} {player['player_id']}**")
                st.metric("Age", f"{int(player['age'])} years")
                st.metric("Overall Score", f"{player['OVERALL SCORE']:.0f}")
                st.metric("Points", f"{player['total_pts']:.1f}")
                st.metric("Toughness", f"{player['mental_toughness']:.1f}/10")
        
        st.markdown("**Full List of Young Players**")
        display_cols = ['player_id', 'age', 'team', 'OVERALL SCORE', 'total_pts', 
                        'total_fg_pct', 'mental_toughness', 'CLUTCH RANK']
        existing = [col for col in display_cols if col in young_players.columns]
        st.dataframe(young_players[existing], use_container_width=True, hide_index=True)
    else:
        st.info("No players under 23 found in this analysis.")
    
    # Age vs Performance Scatter
    st.subheader("📉 Age vs. Performance Correlation")
    
    fig = px.scatter(df_with_age,
                     x='age',
                     y='OVERALL SCORE',
                     color='age_category',
                     size='total_pts',
                     hover_name='player_id',
                     hover_data=['team', 'total_pts', 'mental_toughness'],
                     title="Age vs. Overall Performance Score",
                     labels={'age': 'Age (years)', 'OVERALL SCORE': 'Overall Score'},
                     height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Age vs specific metrics
    st.subheader("📊 Age Impact on Key Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.scatter(df_with_age,
                          x='age',
                          y='mental_toughness',
                          color='age_category',
                          hover_name='player_id',
                          trendline="lowess",
                          title="Age vs. Mental Toughness",
                          labels={'age': 'Age', 'mental_toughness': 'Toughness (1-10)'}
                         )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(df_with_age,
                          x='age',
                          y='total_pts',
                          color='age_category',
                          hover_name='player_id',
                          trendline="lowess",
                          title="Age vs. Total Points",
                          labels={'age': 'Age', 'total_pts': 'Total Points'}
                         )
        st.plotly_chart(fig2, use_container_width=True)

# --- Main Application Logic ---
def main():
    st.title("🏀 AI-Powered Sports Video Analysis")

    # Load data
    age_data = load_age_data()
    data, message = load_latest_analysis_json()

    st.info(message) # Show status message

    if data:
        # Case 1: Classification-only result
        if 'identified_sport' in data and 'players' not in data:
            st.header(f"Sport Identified: {data['identified_sport'].upper()}")
            st.warning("This analysis file only contains sport classification. Run the full two-pass analysis for detailed player stats.")
            st.json(data)
        
        # Case 2: Full Two-Pass analysis result
        elif 'game_info' in data and 'players' in data:
            game_info = data.get('game_info', {})
            video_structure = game_info.get('video_structure', {})
            match_year = video_structure.get('match_year', None)

            # 1. Create base DataFrame
            player_df = create_player_dataframe(data)
            
            if player_df.empty:
                st.error("Failed to parse player data from the JSON file.")
            else:
                # 2. Merge Age Data
                player_df = merge_age_data(player_df, age_data, match_year)
                
                # 3. Generate Gamification Ranks
                player_df_ranked = generate_gamification_rankings(player_df)

                # 4. Add age category to ranked df (if age exists)
                if 'age' in player_df_ranked.columns:
                    player_df_ranked['age_category'] = player_df_ranked['age'].apply(categorize_age)

                # --- Build the Dashboard ---
                
                # Sidebar for navigation
                st.sidebar.title("Navigation")
                page_options = [
                    "Game Overview", 
                    "Player Leaderboard", 
                    "Player Reports", 
                    "Head-to-Head", 
                    "Pressure Analysis",
                    "Age Analysis",
                    "Raw Data"
                ]
                page = st.sidebar.radio("Go to", page_options)
                
                st.sidebar.info("This dashboard visualizes the output of the two-pass AI video analysis.")

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

                elif page == "Raw Data":
                    st.header("Raw JSON Output")
                    st.json(data)
                    st.header("Processed Player DataFrame")
                    st.dataframe(player_df_ranked)

        # Case 3: Unrecognized format
        else:
            st.error("The loaded JSON file is in an unrecognized format.")
            st.json(data) # Show the data for debugging
    else:
        # Case 4: No data found at all
        st.warning("Could not load any analysis data. Please run the analysis script (main.py) first and ensure output files are in the 'output' directory.")
        st.info(f"Looking for files in: {os.path.abspath(OUTPUT_DIR)}")
        st.info(f"Looking for age data at: {os.path.abspath(AGE_DATA_FILE)}")


if __name__ == "__main__":
    main()