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

OUTPUT_DIR = "outputtennis"

st.set_page_config(layout="wide", page_title="🎾 Tennis Analysis Dashboard")

def load_latest_tennis_json():
    """Finds and loads the latest tennis analysis JSON file."""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return None, "Output directory not found. Please run analysis first."

    json_files = glob.glob(str(output_path / "*.json"))
    
    if not json_files:
        return None, "No tennis analysis JSON files found. Run analysis first."

    latest_file = max(json_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return data, f"Successfully loaded: {Path(latest_file).name}"
    except Exception as e:
        return None, f"Error loading file: {e}"

def is_doubles_match(match_data):
    """Determines if the match is doubles based on match_type."""
    try:
        match_type = match_data.get('match_metadata', {}).get('match_type', 'singles')
        return match_type.lower() in ['doubles', 'mixed_doubles']
    except:
        return False

def get_player_stats(match_data, player_key):
    """Safely extracts player statistics."""
    try:
        stats = match_data.get('statistics', {}).get(player_key, {})
        return stats
    except:
        return {}

def calculate_singles_ranking_score(stats, pressure_analysis):
    """Calculate comprehensive ranking score for singles players."""
    try:
        scores = {}
        
        # Serve metrics (30%)
        serve = stats.get('serve', {})
        first_serve_pct = serve.get('first_serve_pct', 0) or 0
        first_serve_won = serve.get('first_serve_points_won_pct', 0) or 0
        aces = serve.get('aces', 0) or 0
        double_faults = serve.get('double_faults', 0) or 0
        
        serve_score = (first_serve_pct * 0.3 + first_serve_won * 0.4 + 
                      min(aces * 2, 20) + max(0, 10 - double_faults))
        scores['Serve'] = min(serve_score, 100)
        
        # Return metrics (25%)
        return_stats = stats.get('return', {})
        break_pts_conv = return_stats.get('break_points_converted', 0) or 0
        break_pts_opp = return_stats.get('break_points_opportunities', 0) or 0
        break_conv_pct = (break_pts_conv / break_pts_opp * 100) if break_pts_opp > 0 else 0
        
        return_score = break_conv_pct * 0.6 + min(break_pts_conv * 5, 40)
        scores['Return'] = min(return_score, 100)
        
        # Point winning (20%)
        points = stats.get('points', {})
        winners = points.get('winners', 0) or 0
        unforced = points.get('unforced_errors', 0) or 0
        total_pts = points.get('total_points_won', 0) or 0
        
        winner_error_ratio = (winners / max(unforced, 1)) * 20
        points_score = min(winner_error_ratio + min(total_pts * 0.5, 50), 100)
        scores['Points'] = points_score
        
        # Pressure performance (25%)
        pressure = pressure_analysis or {}
        clutch_rating = pressure.get('clutch_rating', 5) or 5
        mental_strength = pressure.get('mental_strength', 5) or 5
        break_save_pct = pressure.get('break_points', {}).get('save_pct', 0) or 0
        
        pressure_score = (clutch_rating * 6 + mental_strength * 6 + 
                         break_save_pct * 0.28)
        scores['Pressure'] = min(pressure_score, 100)
        
        # Overall score
        weights = {'Serve': 0.30, 'Return': 0.25, 'Points': 0.20, 'Pressure': 0.25}
        overall = sum(scores[k] * weights[k] for k in scores)
        
        return scores, overall
    except Exception as e:
        st.error(f"Error calculating ranking: {e}")
        return {'Serve': 50, 'Return': 50, 'Points': 50, 'Pressure': 50}, 50

def calculate_doubles_ranking_score(stats, pressure_analysis):
    """Calculate comprehensive ranking score for doubles players."""
    try:
        scores = {}
        
        # Team Serve (35%)
        serve = stats.get('serve', {})
        first_serve_won = serve.get('first_serve_points_won_pct', 0) or 0
        service_games_won = serve.get('service_games_won', 0) or 0
        service_games_total = serve.get('service_games_played', 0) or 0
        hold_pct = (service_games_won / service_games_total * 100) if service_games_total > 0 else 0
        
        serve_score = first_serve_won * 0.5 + hold_pct * 0.5
        scores['Team Serve'] = min(serve_score, 100)
        
        # Net Play (30%)
        points = stats.get('points', {})
        net_pts = points.get('net_points', 0) or 0
        net_won = points.get('net_points_won', 0) or 0
        net_pct = (net_won / net_pts * 100) if net_pts > 0 else 0
        
        net_score = net_pct * 0.7 + min(net_won * 2, 30)
        scores['Net Play'] = min(net_score, 100)
        
        # Return & Break (20%)
        return_stats = stats.get('return', {})
        break_conv = return_stats.get('break_points_converted', 0) or 0
        return_games_won = return_stats.get('return_games_won', 0) or 0
        
        return_score = min(break_conv * 10, 60) + min(return_games_won * 8, 40)
        scores['Return'] = min(return_score, 100)
        
        # Partnership (15%)
        pressure = pressure_analysis or {}
        mental_strength = pressure.get('mental_strength', 5) or 5
        
        partnership_score = mental_strength * 10
        scores['Partnership'] = min(partnership_score, 100)
        
        # Overall score
        weights = {'Team Serve': 0.35, 'Net Play': 0.30, 'Return': 0.20, 'Partnership': 0.15}
        overall = sum(scores[k] * weights[k] for k in scores)
        
        return scores, overall
    except Exception as e:
        st.error(f"Error calculating doubles ranking: {e}")
        return {'Team Serve': 50, 'Net Play': 50, 'Return': 50, 'Partnership': 50}, 50

def create_player_dataframe(match_data):
    """Creates DataFrame with player statistics and rankings."""
    try:
        is_doubles = is_doubles_match(match_data)
        players_info = match_data.get('players', {})
        pressure_data = match_data.get('pressure_analysis', {})
        
        rows = []
        
        for player_key in ['player_1', 'player_2']:
            player_info = players_info.get(player_key, {})
            if not player_info.get('name'):
                continue
                
            stats = get_player_stats(match_data, player_key)
            player_pressure = pressure_data.get(player_key, {})
            
            # Calculate ranking
            if is_doubles:
                component_scores, overall_score = calculate_doubles_ranking_score(stats, player_pressure)
            else:
                component_scores, overall_score = calculate_singles_ranking_score(stats, player_pressure)
            
            row = {
                'Player': player_info.get('name', 'Unknown'),
                'Country': player_info.get('country', 'N/A'),
                'Hand': player_info.get('serves', 'N/A'),
                'Overall Score': round(overall_score, 1),
                **{f'{k} Score': round(v, 1) for k, v in component_scores.items()}
            }
            
            # Add key statistics
            serve_stats = stats.get('serve', {})
            row['Aces'] = serve_stats.get('aces', 0) or 0
            row['Double Faults'] = serve_stats.get('double_faults', 0) or 0
            row['1st Serve %'] = round(serve_stats.get('first_serve_pct', 0) or 0, 1)
            row['1st Serve Won %'] = round(serve_stats.get('first_serve_points_won_pct', 0) or 0, 1)
            
            return_stats = stats.get('return', {})
            row['Break Points Won'] = f"{return_stats.get('break_points_converted', 0) or 0}/{return_stats.get('break_points_opportunities', 0) or 0}"
            
            points_stats = stats.get('points', {})
            row['Winners'] = points_stats.get('winners', 0) or 0
            row['Unforced Errors'] = points_stats.get('unforced_errors', 0) or 0
            row['Total Points Won'] = points_stats.get('total_points_won', 0) or 0
            
            # Pressure metrics
            row['Clutch Rating'] = round(player_pressure.get('clutch_rating', 0) or 0, 1)
            row['Mental Strength'] = round(player_pressure.get('mental_strength', 0) or 0, 1)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if not df.empty and 'Overall Score' in df.columns:
            df['Rank'] = rankdata(-df['Overall Score'], method='min').astype(int)
            df = df.sort_values('Rank')
        
        return df, is_doubles
    except Exception as e:
        st.error(f"Error creating player dataframe: {e}")
        return pd.DataFrame(), False

def display_match_overview(match_data):
    """Display match overview and key information."""
    st.header("🎾 Match Overview")
    
    try:
        metadata = match_data.get('match_metadata', {})
        result = match_data.get('match_result', {})
        players = match_data.get('players', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tournament = metadata.get('tournament', 'N/A')
            st.metric("Tournament", tournament)
            
        with col2:
            round_info = metadata.get('round', 'N/A')
            st.metric("Round", round_info)
            
        with col3:
            surface = metadata.get('surface', 'N/A')
            st.metric("Surface", surface.title())
            
        with col4:
            match_type = metadata.get('match_type', 'singles')
            st.metric("Match Type", match_type.replace('_', ' ').title())
        
        # Players and Score
        st.subheader("Match Result")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            p1 = players.get('player_1', {})
            st.markdown(f"### {p1.get('name', 'Player 1')}")
            st.caption(f"🏴 {p1.get('country', 'N/A')} | {p1.get('serves', 'N/A')}-handed")
            
        with col2:
            score = result.get('score', [])
            if score:
                st.markdown("### Score")
                for set_score in score:
                    st.markdown(f"**{set_score}**")
            else:
                st.markdown("### vs")
                
        with col3:
            p2 = players.get('player_2', {})
            st.markdown(f"### {p2.get('name', 'Player 2')}")
            st.caption(f"🏴 {p2.get('country', 'N/A')} | {p2.get('serves', 'N/A')}-handed")
        
        # Match Statistics
        st.subheader("Match Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        totals = match_data.get('statistics', {}).get('match_totals', {})
        
        with col1:
            st.metric("Total Games", totals.get('total_games', 0))
        with col2:
            st.metric("Total Points", totals.get('total_points', 0))
        with col3:
            st.metric("Total Aces", totals.get('total_aces', 0))
        with col4:
            duration = metadata.get('duration_seconds', 0)
            if duration:
                mins = duration // 60
                st.metric("Duration", f"{mins} min")
            else:
                st.metric("Duration", "N/A")
                
    except Exception as e:
        st.error(f"Error displaying match overview: {e}")

def display_singles_ranking(df):
    """Display singles player ranking with detailed breakdown."""
    st.header("🏆 Singles Player Ranking")
    
    try:
        if df.empty:
            st.warning("No player data available.")
            return
        
        # Main ranking table
        display_cols = ['Rank', 'Player', 'Country', 'Overall Score', 
                       'Serve Score', 'Return Score', 'Points Score', 'Pressure Score',
                       'Aces', '1st Serve %', 'Winners', 'Clutch Rating']
        
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div style='font-size: small; color: grey; margin-top: 10px;'>
            **Overall Score** is calculated from: Serve (30%), Return (25%), Points (20%), and Pressure Performance (25%)
        </div>
        """, unsafe_allow_html=True)
        
        # Score breakdown visualization
        st.subheader("Score Component Breakdown")
        
        score_cols = [col for col in df.columns if 'Score' in col and col != 'Overall Score']
        
        if score_cols:
            fig = go.Figure()
            
            for _, row in df.iterrows():
                fig.add_trace(go.Bar(
                    name=row['Player'],
                    x=score_cols,
                    y=[row[col] for col in score_cols],
                    text=[f"{row[col]:.1f}" for col in score_cols],
                    textposition='auto',
                ))
            
            fig.update_layout(
                barmode='group',
                title="Component Scores Comparison",
                xaxis_title="Score Components",
                yaxis_title="Score (0-100)",
                yaxis=dict(range=[0, 100]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying singles ranking: {e}")

def display_doubles_ranking(df):
    """Display doubles team ranking with specialized metrics."""
    st.header("🏆 Doubles Team Ranking")
    
    try:
        if df.empty:
            st.warning("No player data available.")
            return
        
        # Main ranking table
        display_cols = ['Rank', 'Player', 'Country', 'Overall Score',
                       'Team Serve Score', 'Net Play Score', 'Return Score', 'Partnership Score',
                       'Aces', '1st Serve Won %', 'Winners', 'Mental Strength']
        
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div style='font-size: small; color: grey; margin-top: 10px;'>
            **Overall Score** for Doubles is based on: Team Serve (35%), Net Play (30%), Return (20%), and Partnership (15%)
        </div>
        """, unsafe_allow_html=True)
        
        # Score breakdown
        st.subheader("Doubles Performance Breakdown")
        
        score_cols = [col for col in df.columns if 'Score' in col and col != 'Overall Score']
        
        if score_cols:
            fig = go.Figure()
            
            for _, row in df.iterrows():
                fig.add_trace(go.Bar(
                    name=row['Player'],
                    x=score_cols,
                    y=[row[col] for col in score_cols],
                    text=[f"{row[col]:.1f}" for col in score_cols],
                    textposition='auto',
                ))
            
            fig.update_layout(
                barmode='group',
                title="Doubles Component Scores",
                xaxis_title="Score Components",
                yaxis_title="Score (0-100)",
                yaxis=dict(range=[0, 100]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying doubles ranking: {e}")

def display_detailed_statistics(df, match_data):
    """Display detailed player statistics and comparisons."""
    st.header("📊 Detailed Statistics")
    
    try:
        if df.empty:
            st.warning("No player statistics available.")
            return
        
        # Serve statistics
        st.subheader("Serve Performance")
        
        serve_metrics = ['Player', 'Aces', 'Double Faults', '1st Serve %', '1st Serve Won %']
        available_serve = [col for col in serve_metrics if col in df.columns]
        
        if available_serve:
            st.dataframe(df[available_serve], use_container_width=True, hide_index=True)
            
            # Serve visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Aces',
                x=df['Player'],
                y=df['Aces'],
                marker_color='green'
            ))
            fig.add_trace(go.Bar(
                name='Double Faults',
                x=df['Player'],
                y=df['Double Faults'],
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Aces vs Double Faults",
                barmode='group',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Points statistics
        st.subheader("Point Breakdown")
        
        points_metrics = ['Player', 'Winners', 'Unforced Errors', 'Total Points Won']
        available_points = [col for col in points_metrics if col in df.columns]
        
        if available_points:
            st.dataframe(df[available_points], use_container_width=True, hide_index=True)
            
            # Winner/Error ratio
            fig = px.bar(df, x='Player', y=['Winners', 'Unforced Errors'],
                        title="Winners vs Unforced Errors",
                        barmode='group',
                        height=350,
                        color_discrete_map={'Winners': 'lightgreen', 'Unforced Errors': 'lightcoral'})
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying detailed statistics: {e}")

def display_pressure_analysis(df, match_data):
    """Display pressure and clutch performance analysis."""
    st.header("🔥 Pressure Performance Analysis")
    
    try:
        if df.empty:
            st.warning("No pressure data available.")
            return
        
        pressure_data = match_data.get('pressure_analysis', {})
        
        # Clutch performance metrics
        st.subheader("Clutch Performance Metrics")
        
        clutch_cols = ['Player', 'Clutch Rating', 'Mental Strength', 'Break Points Won']
        available_clutch = [col for col in clutch_cols if col in df.columns]
        
        if available_clutch:
            st.dataframe(df[available_clutch], use_container_width=True, hide_index=True)
        
        # Radar chart for pressure metrics
        if 'Clutch Rating' in df.columns and 'Mental Strength' in df.columns:
            st.subheader("Pressure Performance Comparison")
            
            fig = go.Figure()
            
            categories = ['Clutch Rating', 'Mental Strength', 'Overall Score']
            
            for _, row in df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[
                        row.get('Clutch Rating', 0) * 10,
                        row.get('Mental Strength', 0) * 10,
                        row.get('Overall Score', 0)
                    ],
                    theta=categories,
                    fill='toself',
                    name=row['Player']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                height=500,
                title="Mental & Pressure Performance Radar"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Key moments
        st.subheader("⚡ Key Pressure Moments")
        
        key_moments = pressure_data.get('key_moments', [])
        if key_moments:
            moments_df = pd.DataFrame(key_moments)
            display_moment_cols = [col for col in ['timestamp', 'set', 'game', 'description', 'pressure_level', 'outcome'] 
                                  if col in moments_df.columns]
            if display_moment_cols:
                st.dataframe(moments_df[display_moment_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No key pressure moments recorded.")
            
    except Exception as e:
        st.error(f"Error displaying pressure analysis: {e}")

def display_match_momentum(match_data):
    """Display match momentum and key events."""
    st.header("📈 Match Momentum & Key Events")
    
    try:
        events = match_data.get('events', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Service Breaks")
            breaks = events.get('breaks_of_serve', [])
            if breaks:
                breaks_df = pd.DataFrame(breaks)
                st.dataframe(breaks_df, use_container_width=True, hide_index=True)
            else:
                st.info("No service breaks recorded.")
        
        with col2:
            st.subheader("Pressure Points")
            pressure_points = events.get('pressure_points', [])
            if pressure_points:
                pressure_df = pd.DataFrame(pressure_points)
                st.metric("Total Pressure Points", len(pressure_points))
                
                # Pressure level distribution
                if 'pressure_level' in pressure_df.columns:
                    avg_pressure = pressure_df['pressure_level'].mean()
                    st.metric("Average Pressure Level", f"{avg_pressure:.1f}/10")
            else:
                st.info("No pressure points recorded.")
        
        # Momentum shifts
        st.subheader("Momentum Shifts")
        momentum_shifts = events.get('momentum_shifts', [])
        if momentum_shifts:
            momentum_df = pd.DataFrame(momentum_shifts)
            st.dataframe(momentum_df, use_container_width=True, hide_index=True)
        else:
            st.info("No momentum shifts recorded.")
            
    except Exception as e:
        st.error(f"Error displaying match momentum: {e}")

def display_head_to_head(df):
    """Display direct comparison between players."""
    st.header("⚔️ Head-to-Head Comparison")
    
    try:
        if df.empty or len(df) < 2:
            st.info("Need at least 2 players for comparison.")
            return
        
        players = df['Player'].tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            player1 = st.selectbox("Select Player 1", players, key='p1')
        
        with col2:
            player2 = st.selectbox("Select Player 2", players, key='p2')
        
        if player1 and player2 and player1 != player2:
            p1_data = df[df['Player'] == player1].iloc[0]
            p2_data = df[df['Player'] == player2].iloc[0]
            
            # Comparison metrics
            st.subheader(f"{player1} vs {player2}")
            
            comparison_metrics = ['Overall Score', 'Aces', 'Winners', '1st Serve %', 
                                'Clutch Rating', 'Mental Strength']
            
            comp_data = []
            for metric in comparison_metrics:
                if metric in df.columns:
                    comp_data.append({
                        'Metric': metric,
                        player1: p1_data.get(metric, 0),
                        player2: p2_data.get(metric, 0)
                    })
            
            if comp_data:
                comp_df = pd.DataFrame(comp_data).set_index('Metric')
                st.dataframe(comp_df, use_container_width=True)
                
                # Visual comparison
                fig = px.bar(comp_df.reset_index().melt('Metric', var_name='Player', value_name='Value'),
                           x='Metric', y='Value', color='Player', barmode='group',
                           title="Key Metrics Comparison",
                           height=400)
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in head-to-head comparison: {e}")

# Main Dashboard
def main():
    match_data, status_message = load_latest_tennis_json()
    
    st.title("🎾 Tennis Match Analysis Dashboard")
    st.caption(f"Status: {status_message}")
    
    if match_data is None:
        st.error("Cannot load analysis data. Please check the outputtennis directory.")
        return
    
    try:
        player_df, is_doubles = create_player_dataframe(match_data)
        
        # Create tabs based on match type
        if is_doubles:
            tabs = st.tabs([
                "Match Overview",
                "Doubles Ranking",
                "Detailed Statistics",
                "Pressure Analysis",
                "Match Momentum",
                "Head-to-Head"
            ])
            
            with tabs[0]:
                display_match_overview(match_data)
            
            with tabs[1]:
                display_doubles_ranking(player_df)
            
            with tabs[2]:
                display_detailed_statistics(player_df, match_data)
            
            with tabs[3]:
                display_pressure_analysis(player_df, match_data)
            
            with tabs[4]:
                display_match_momentum(match_data)
            
            with tabs[5]:
                display_head_to_head(player_df)
        else:
            tabs = st.tabs([
                "Match Overview",
                "Singles Ranking",
                "Detailed Statistics",
                "Pressure Analysis",
                "Match Momentum",
                "Head-to-Head"
            ])
            
            with tabs[0]:
                display_match_overview(match_data)
            
            with tabs[1]:
                display_singles_ranking(player_df)
            
            with tabs[2]:
                display_detailed_statistics(player_df, match_data)
            
            with tabs[3]:
                display_pressure_analysis(player_df, match_data)
            
            with tabs[4]:
                display_match_momentum(match_data)
            
            with tabs[5]:
                display_head_to_head(player_df)
                
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        st.exception(e)
        
        # Show raw JSON for debugging
        with st.expander("View Raw JSON Data"):
            st.json(match_data)

if __name__ == "__main__":
    main()