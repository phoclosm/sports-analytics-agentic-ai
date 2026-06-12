import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="⚽ Football Match Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .trophy-gold { color: #ffd700; font-size: 2rem; }
    .trophy-silver { color: #c0c0c0; font-size: 1.8rem; }
    .trophy-bronze { color: #cd7f32; font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_latest_match():
    """Load the latest JSON file from outputfoot folder."""
    output_dir = Path("outputfoot")
    if not output_dir.exists():
        st.error("❌ outputfoot folder not found!")
        return None
    
    json_files = list(output_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.stem.endswith(('_players', '_events'))]
    
    if not json_files:
        st.error("❌ No match JSON files found in outputfoot folder!")
        return None
    
    latest_file = max(json_files, key=os.path.getmtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data, latest_file.stem


def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """Safely convert value to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_str(value, default=""):
    """Safely convert value to string."""
    try:
        if value is None:
            return default
        return str(value).replace("'", "")
    except:
        return default


def get_rating_color(rating):
    """Get color based on rating."""
    if rating >= 8.0:
        return "#10b981"
    elif rating >= 7.0:
        return "#3b82f6"
    elif rating >= 6.0:
        return "#f59e0b"
    else:
        return "#ef4444"


def create_player_radar(player):
    """Create radar chart for player performance."""
    categories = ['Goals', 'Assists', 'Shots', 'Pass Acc', 'Tackles', 'Interceptions']
    
    perf = player['performance']
    
    values = [
        min(safe_float(perf.get('goals', 0)) * 2, 10),
        min(safe_float(perf.get('assists', 0)) * 2, 10),
        min(safe_float(perf.get('shots', 0)) / 2, 10),
        safe_float(perf.get('pass_accuracy', 0)) / 10,
        min(safe_float(perf.get('tackles', 0)), 10),
        min(safe_float(perf.get('interceptions', 0)), 10)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#3b82f6',
        fillcolor='rgba(59, 130, 246, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig


def create_pressure_timeline(match_data):
    """Create timeline of pressure moments."""
    pressure_moments = match_data['events']['pressure_moments']
    
    if not pressure_moments:
        return None
    
    df = pd.DataFrame(pressure_moments)
    df['minute_num'] = df['minute'].apply(lambda x: safe_float(x))
    df['intensity_num'] = df['intensity'].apply(lambda x: safe_float(x))
    
    fig = go.Figure()
    colors = df['intensity_num'].apply(lambda x: '#ef4444' if x >= 8 else '#f59e0b' if x >= 6 else '#10b981')
    
    fig.add_trace(go.Scatter(
        x=df['minute_num'],
        y=df['intensity_num'],
        mode='markers+lines',
        marker=dict(size=df['intensity_num'] * 3, color=colors, line=dict(width=2, color='white')),
        line=dict(color='#94a3b8', width=1),
        text=df['type'],
        hovertemplate='<b>%{text}</b><br>Minute: %{x}<br>Intensity: %{y}/10<extra></extra>'
    ))
    
    fig.update_layout(
        title="Match Pressure Timeline",
        xaxis_title="Match Minute",
        yaxis_title="Pressure Intensity",
        yaxis=dict(range=[0, 11]),
        height=400,
        hovermode='closest'
    )
    
    return fig


def create_score_flow(match_data):
    """Create score progression chart."""
    goals = match_data['events']['goals']
    
    if not goals:
        return None
    
    timeline = [{'minute': 0, 'home': 0, 'away': 0}]
    
    for goal in goals:
        minute = safe_float(safe_str(goal.get('minute', 0)))
        timeline.append({
            'minute': minute,
            'home': safe_int(goal['score_after']['home']),
            'away': safe_int(goal['score_after']['away'])
        })
    
    df = pd.DataFrame(timeline)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['minute'], y=df['home'],
        name=match_data['teams']['home']['name'],
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['minute'], y=df['away'],
        name=match_data['teams']['away']['name'],
        mode='lines+markers',
        line=dict(color='#ef4444', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Score Progression",
        xaxis_title="Match Minute",
        yaxis_title="Goals",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def main():
    st.markdown('<h1 class="main-header">⚽ Football Match Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    result = load_latest_match()
    if result is None:
        return
    
    match_data, match_id = result
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/football2--v1.png", width=100)
        st.title("Match Info")
        st.markdown("---")
        
        home_team = match_data['teams']['home']['name']
        away_team = match_data['teams']['away']['name']
        home_score = safe_int(match_data['teams']['home']['score'])
        away_score = safe_int(match_data['teams']['away']['score'])
        
        st.subheader(f"{home_team} vs {away_team}")
        st.metric("Final Score", f"{home_score} - {away_score}")
        
        st.markdown("---")
        st.info(f"**Competition:** {match_data['match_metadata']['competition']}")
        if match_data['match_metadata'].get('date'):
            st.info(f"**Date:** {match_data['match_metadata']['date']}")
        if match_data['match_metadata'].get('venue'):
            st.info(f"**Venue:** {match_data['match_metadata']['venue']}")
        
        st.markdown("---")
        st.caption(f"Schema: v{match_data['schema_version']}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Leaderboard", "⚔️ Head-to-Head", "📊 Player Reports",
        "📈 Match Overview", "🧠 Pressure Analysis"
    ])
    
    players = match_data['players']
    
    # TAB 1: LEADERBOARD
    with tab1:
        st.header("🏆 Player Leaderboard")
        
        sorted_players = sorted(players, key=lambda x: safe_float(x['performance']['rating']), reverse=True)
        
        # Top 3
        col1, col2, col3 = st.columns(3)
        
        if len(sorted_players) >= 1:
            with col2:
                p = sorted_players[0]
                st.markdown('<div style="text-align: center;"><span class="trophy-gold">🥇</span></div>', unsafe_allow_html=True)
                st.markdown(f"### {p['name']}")
                st.metric("Rating", f"{safe_float(p['performance']['rating']):.1f}/10")
                st.caption(f"#{p['jersey_number']} • {p['team']}")
        
        if len(sorted_players) >= 2:
            with col1:
                p = sorted_players[1]
                st.markdown('<div style="text-align: center;"><span class="trophy-silver">🥈</span></div>', unsafe_allow_html=True)
                st.markdown(f"### {p['name']}")
                st.metric("Rating", f"{safe_float(p['performance']['rating']):.1f}/10")
                st.caption(f"#{p['jersey_number']} • {p['team']}")
        
        if len(sorted_players) >= 3:
            with col3:
                p = sorted_players[2]
                st.markdown('<div style="text-align: center;"><span class="trophy-bronze">🥉</span></div>', unsafe_allow_html=True)
                st.markdown(f"### {p['name']}")
                st.metric("Rating", f"{safe_float(p['performance']['rating']):.1f}/10")
                st.caption(f"#{p['jersey_number']} • {p['team']}")
        
        st.markdown("---")
        st.subheader("📋 Full Rankings")
        
        leaderboard_data = []
        for idx, p in enumerate(sorted_players, 1):
            perf = p['performance']
            
            leaderboard_data.append({
                'Rank': idx,
                'Medal': '🥇' if idx == 1 else '🥈' if idx == 2 else '🥉' if idx == 3 else '',
                'Player': p['name'],
                'Jersey': safe_int(p['jersey_number']),
                'Team': p['team'],
                'Rating': safe_float(perf['rating']),
                'Goals': safe_int(perf['goals']),
                'Assists': safe_int(perf['assists']),
                'Shots': safe_int(perf['shots']),
                'Pass%': safe_float(perf['pass_accuracy']),
                'Tackles': safe_int(perf['tackles']),
                'Minutes': safe_int(p['minutes_played'])
            })
        
        df_leaderboard = pd.DataFrame(leaderboard_data)
        st.dataframe(df_leaderboard, hide_index=True, height=600)
        
        # Category leaders
        st.markdown("---")
        st.subheader("🎖️ Category Leaders")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            top_scorer = max(players, key=lambda x: safe_int(x['performance']['goals']))
            st.metric("⚽ Top Scorer", top_scorer['name'], f"{safe_int(top_scorer['performance']['goals'])} goals")
        
        with col2:
            top_assister = max(players, key=lambda x: safe_int(x['performance']['assists']))
            st.metric("🎯 Top Assister", top_assister['name'], f"{safe_int(top_assister['performance']['assists'])} assists")
        
        with col3:
            top_passer = max(players, key=lambda x: safe_float(x['performance']['pass_accuracy']) if safe_int(x['performance']['passes']) > 0 else 0)
            st.metric("✅ Best Passer", top_passer['name'], f"{safe_float(top_passer['performance']['pass_accuracy']):.0f}%")
        
        with col4:
            top_defender = max(players, key=lambda x: safe_int(x['performance']['tackles']))
            st.metric("🛡️ Top Defender", top_defender['name'], f"{safe_int(top_defender['performance']['tackles'])} tackles")
    
    # TAB 2: HEAD-TO-HEAD
    with tab2:
        st.header("⚔️ Head-to-Head Comparison")
        
        player_names = [p['name'] for p in players]
        
        col1, col2 = st.columns(2)
        with col1:
            player1_name = st.selectbox("Player 1", player_names, key='p1')
        with col2:
            player2_name = st.selectbox("Player 2", player_names, index=min(1, len(player_names)-1), key='p2')
        
        player1 = next(p for p in players if p['name'] == player1_name)
        player2 = next(p for p in players if p['name'] == player2_name)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {player1['name']}")
            st.caption(f"#{player1['jersey_number']} • {player1['team']} • {player1['position']}")
            perf1 = player1['performance']
            st.metric("Rating", f"{safe_float(perf1['rating']):.1f}/10")
            st.text(f"Goals: {safe_int(perf1['goals'])} | Assists: {safe_int(perf1['assists'])}")
            st.text(f"Shots: {safe_int(perf1['shots'])} | Passes: {safe_int(perf1['passes'])}")
            st.text(f"Pass Accuracy: {safe_float(perf1['pass_accuracy']):.1f}%")
            st.text(f"Tackles: {safe_int(perf1['tackles'])} | Interceptions: {safe_int(perf1['interceptions'])}")
        
        with col2:
            st.markdown(f"### {player2['name']}")
            st.caption(f"#{player2['jersey_number']} • {player2['team']} • {player2['position']}")
            perf2 = player2['performance']
            st.metric("Rating", f"{safe_float(perf2['rating']):.1f}/10")
            st.text(f"Goals: {safe_int(perf2['goals'])} | Assists: {safe_int(perf2['assists'])}")
            st.text(f"Shots: {safe_int(perf2['shots'])} | Passes: {safe_int(perf2['passes'])}")
            st.text(f"Pass Accuracy: {safe_float(perf2['pass_accuracy']):.1f}%")
            st.text(f"Tackles: {safe_int(perf2['tackles'])} | Interceptions: {safe_int(perf2['interceptions'])}")
        
        st.markdown("---")
        st.subheader("📊 Performance Comparison")
        
        categories = ['Goals', 'Assists', 'Shots', 'Pass Acc', 'Tackles', 'Interceptions']
        
        values1 = [
            min(safe_float(perf1['goals']) * 2, 10),
            min(safe_float(perf1['assists']) * 2, 10),
            min(safe_float(perf1['shots']) / 2, 10),
            safe_float(perf1['pass_accuracy']) / 10,
            min(safe_float(perf1['tackles']), 10),
            min(safe_float(perf1['interceptions']), 10)
        ]
        
        values2 = [
            min(safe_float(perf2['goals']) * 2, 10),
            min(safe_float(perf2['assists']) * 2, 10),
            min(safe_float(perf2['shots']) / 2, 10),
            safe_float(perf2['pass_accuracy']) / 10,
            min(safe_float(perf2['tackles']), 10),
            min(safe_float(perf2['interceptions']), 10)
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values1, theta=categories, fill='toself', name=player1['name'], line_color='#3b82f6'))
        fig.add_trace(go.Scatterpolar(r=values2, theta=categories, fill='toself', name=player2['name'], line_color='#ef4444'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: PLAYER REPORTS
    with tab3:
        st.header("📊 Player Report Cards")
        
        player_names = [p['name'] for p in players]
        selected_name = st.selectbox("Select Player", player_names)
        
        player = next(p for p in players if p['name'] == selected_name)
        perf = player['performance']
        
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.markdown(f"<div style='text-align: center; font-size: 4rem;'>#{player['jersey_number']}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### {player['name']}")
            st.caption(f"**{player['team']}** • {player['position']}")
            st.caption(f"Minutes: {safe_int(player['minutes_played'])}")
        
        with col3:
            rating_color = get_rating_color(safe_float(perf['rating']))
            st.markdown(f"<div style='text-align: center;'><h1 style='color: {rating_color};'>{safe_float(perf['rating']):.1f}</h1><p>Overall Rating</p></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Performance Profile")
            st.plotly_chart(create_player_radar(player), use_container_width=True)
        
        with col2:
            st.subheader("Statistics")
            st.metric("Goals", safe_int(perf['goals']))
            st.metric("Assists", safe_int(perf['assists']))
            st.metric("Shots", safe_int(perf['shots']))
            st.metric("Pass Accuracy", f"{safe_float(perf['pass_accuracy']):.1f}%")
            st.metric("Tackles", safe_int(perf['tackles']))
            st.metric("Interceptions", safe_int(perf['interceptions']))
    
    # TAB 4: MATCH OVERVIEW
    with tab4:
        st.header("📈 Match Overview")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"<h2 style='text-align: right;'>{home_team}</h2>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h1 style='text-align: center; color: #3b82f6;'>{home_score} - {away_score}</h1>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<h2 style='text-align: left;'>{away_team}</h2>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        score_fig = create_score_flow(match_data)
        if score_fig:
            st.plotly_chart(score_fig, use_container_width=True)
        
        st.subheader("⚡ Match Events")
        
        if match_data['events']['goals']:
            st.markdown("**⚽ Goals**")
            for goal in match_data['events']['goals']:
                assist = f" (Assist: {goal['assist_player_name']})" if goal.get('assist_player_name') else ""
                st.text(f"{safe_str(goal['minute'])}' - {goal['player_name']} - {goal['type']}{assist} [{safe_int(goal['score_after']['home'])}-{safe_int(goal['score_after']['away'])}]")
        
        if match_data['events']['cards']:
            st.markdown("**🟨🟥 Cards**")
            for card in match_data['events']['cards']:
                emoji = "🟨" if card['card_type'] == 'yellow' else "🟥"
                st.text(f"{safe_str(card['minute'])}' - {emoji} {card['player_name']} ({card['team']})")
        
        if match_data['events']['substitutions']:
            st.markdown("**🔄 Substitutions**")
            for sub in match_data['events']['substitutions']:
                st.text(f"{safe_str(sub['minute'])}' - {sub['player_off_name']} ➡️ {sub['player_on_name']} ({sub['team']})")
        
        st.markdown("---")
        totals = match_data['statistics']['match_totals']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Goals", safe_int(totals['total_goals']))
        with col2:
            st.metric("Fouls", safe_int(totals['total_fouls']))
        with col3:
            st.metric("Cards", safe_int(totals['total_cards']))
        with col4:
            st.metric("Subs", safe_int(totals['total_substitutions']))
        with col5:
            st.metric("Players", safe_int(totals['total_players']))
    
    # TAB 5: PRESSURE ANALYSIS
    with tab5:
        st.header("🧠 Mental & Pressure Analysis")
        
        pressure_moments = match_data['events']['pressure_moments']
        
        if not pressure_moments:
            st.info("No pressure moments detected in this match.")
        else:
            totals = match_data['statistics']['match_totals']
            avg_intensity = safe_float(totals.get('avg_pressure_intensity', 0))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Intensity", f"{avg_intensity:.1f}/10")
            with col2:
                st.metric("Total Moments", len(pressure_moments))
            with col3:
                critical = len([pm for pm in pressure_moments if safe_float(pm['intensity']) >= 8])
                st.metric("Critical (8+)", critical)
            
            st.markdown("---")
            
            pressure_fig = create_pressure_timeline(match_data)
            if pressure_fig:
                st.plotly_chart(pressure_fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔥 Pressure Moments")
            
            for pm in pressure_moments:
                intensity = safe_float(pm['intensity'])
                emoji = "🔴" if intensity >= 8 else "🟠" if intensity >= 6 else "🟢"
                label = "CRITICAL" if intensity >= 8 else "HIGH" if intensity >= 6 else "MEDIUM"
                
                with st.expander(f"{emoji} {safe_str(pm['minute'])}' - {pm['type']} ({label})"):
                    st.text(f"Intensity: {intensity:.0f}/10")
                    st.text(f"Team: {pm['team_under_pressure']}")
                    st.text(f"Outcome: {pm['outcome']}")
                    score = pm['score_situation']
                    st.text(f"Score: {safe_int(score['home'])}-{safe_int(score['away'])}")


if __name__ == "__main__":
    main()