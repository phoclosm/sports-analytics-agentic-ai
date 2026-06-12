from google import genai
import pandas as pd
import json
from datetime import datetime
import os
import logging
import time
import asyncio
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoPassFootballAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.skim_model_name = 'models/gemini-2.5-flash'
        self.focus_model_name = 'models/gemini-2.5-flash'
    
    def create_skim_prompt(self) -> str:
        return """PASS 1: COMPREHENSIVE MATCH STRUCTURE ANALYSIS

CRITICAL INSTRUCTIONS FOR TEAM & PLAYER IDENTIFICATION:

1. TEAM IDENTIFICATION STRATEGY (Multi-source verification):
   - Primary: Check SCOREBOARD graphics (shows team names and home/away designation)
   - Secondary: Observe GOAL CELEBRATIONS (celebrating team = scoring team)
   - Tertiary: Track JERSEY PATTERNS consistently throughout video
   - Quaternary: Listen to COMMENTARY for team name mentions
   - Cross-reference ALL sources before assigning teams

2. PLAYER IDENTIFICATION PROTOCOL:
   - Watch REPLAYS carefully for clear jersey numbers
   - Check on-screen GRAPHICS showing player names after goals/key moments
   - Verify player belongs to correct team via jersey pattern matching
   - Use SCOREBOARD overlays that show player names
   - Only include players you can CONFIDENTLY identify with number AND name
   - If uncertain about player identity, mark as "Unknown Player #XX"

3. JERSEY PATTERN RECOGNITION (Not just colors):
   - Note STRIPE PATTERNS (vertical, horizontal, diagonal)
   - Observe SPONSOR LOGOS and placement
   - Track SHORTS COLOR in addition to jersey
   - Identify GOALKEEPER distinct kit
   - Look for SECONDARY/AWAY kit if team changes jerseys

4. GOAL VALIDATION RULES:
   - Scorer MUST be visible in replay
   - Jersey number MUST match jersey pattern of scoring team
   - Cross-check scoreboard update with visual confirmation
   - If goal scorer unclear, mark scorer as "Unknown Player"

OUTPUT STRUCTURE (VALID JSON ONLY):
{
  "video_structure": {
    "team_home": "TEAM NAME",
    "team_away": "TEAM NAME",
    "team_home_jersey": {
      "primary_colors": ["Color1", "Color2"],
      "pattern": "solid/striped/checkered",
      "sponsor": "Sponsor Name",
      "shorts_color": "Color"
    },
    "team_away_jersey": {
      "primary_colors": ["Color1", "Color2"],
      "pattern": "solid/striped/checkered",
      "sponsor": "Sponsor Name",
      "shorts_color": "Color"
    },
    "competition": "League/Tournament Name",
    "match_date_visible": "YYYY-MM-DD or Unknown",
    "venue": "Stadium Name or Unknown",
    "total_duration_seconds": 0,
    "periods_detected": ["First Half", "Second Half"],
    "stoppage_time_added": {
      "first_half": 0,
      "second_half": 0
    },
    "initial_score": {"home": 0, "away": 0},
    "final_score": {"home": 0, "away": 0},
    "match_officials": {
      "referee_visible": true,
      "var_usage_detected": false
    }
  },
  "players_detected": [
    {
      "player_id": "#7",
      "player_name": "Player Name",
      "team": "TEAM NAME",
      "jersey_pattern": {
        "colors": ["Color1"],
        "pattern": "solid",
        "shorts_color": "Color"
      },
      "position": "Forward/Midfielder/Defender/Goalkeeper",
      "visible_periods": ["First Half", "Second Half"],
      "is_starter": true,
      "estimated_minutes": 90,
      "confidence_level": "high/medium/low"
    }
  ],
  "goals_scored": [
    {
      "goal_number": 1,
      "timestamp": "0:36",
      "game_minute": "12'",
      "period": "First Half",
      "scorer_jersey_number": "#7",
      "scorer_name": "Player Name",
      "scoring_team": "TEAM NAME",
      "assist_by_jersey": "#10",
      "assist_by_name": "Player Name",
      "goal_type": "open_play/penalty/free_kick/corner/header/counter_attack",
      "score_before": {"home": 0, "away": 0},
      "score_after": {"home": 1, "away": 0},
      "is_equalizer": false,
      "is_winning_goal": false,
      "is_go_ahead_goal": false,
      "body_part": "right_foot/left_foot/header/chest",
      "distance_from_goal_meters": 0,
      "buildup_passes": 0,
      "time_from_possession_seconds": 0
    }
  ],
  "fouls_and_cards": [
    {
      "timestamp": "1:23",
      "game_minute": "45'",
      "period": "First Half",
      "foul_by_jersey": "#5",
      "foul_by_name": "Player Name",
      "foul_by_team": "TEAM NAME",
      "foul_on_jersey": "#7",
      "foul_on_name": "Player Name",
      "foul_type": "tactical/dangerous/professional",
      "card_given": "yellow/red/none",
      "free_kick_awarded": true,
      "location_on_pitch": "defensive_third/middle_third/attacking_third"
    }
  ],
  "set_pieces": [
    {
      "timestamp": "2:15",
      "game_minute": "67'",
      "type": "corner/free_kick/penalty/throw_in",
      "taken_by_team": "TEAM NAME",
      "taker_jersey": "#10",
      "taker_name": "Player Name",
      "outcome": "goal/shot_saved/cleared/out_of_play",
      "resulted_in_goal": false
    }
  ],
  "substitutions": [
    {
      "timestamp": "3:05",
      "game_minute": "78'",
      "team": "TEAM NAME",
      "player_off_jersey": "#9",
      "player_off_name": "Player Name",
      "player_on_jersey": "#21",
      "player_on_name": "Player Name",
      "reason_visible": "tactical/injury/time_wasting"
    }
  ],
  "key_moments": [
    {
      "timestamp": "0:20-0:40",
      "period": "First Half",
      "moment_type": "goal/near_miss/save/controversial_decision/injury",
      "importance": "critical/high/medium",
      "reason": "Description of moment",
      "current_score": {"home": 0, "away": 0},
      "key_players_involved": ["#7 Player Name", "#10 Player Name"],
      "match_context": "close_game/one_team_dominating/comeback_attempt"
    }
  ],
  "pressure_moments": [
    {
      "timestamp": "3:45-3:50",
      "game_minute": "88'-90'",
      "period": "Second Half",
      "pressure_type": "late_game_drama/penalty_situation/red_card_impact/injury_time",
      "score_situation": {"home": 2, "away": 2},
      "team_under_pressure": "TEAM NAME",
      "intensity_rating": 9,
      "outcome": "goal_scored/defended_successfully/opportunity_missed"
    }
  ],
  "tactical_observations": [
    {
      "period": "First Half",
      "team": "TEAM NAME",
      "formation_detected": "4-3-3/4-4-2/3-5-2",
      "playing_style": "possession/counter_attack/direct/pressing",
      "key_tactical_changes": "Dropped deeper after leading"
    }
  ],
  "focus_segments": [
    {
      "start_time": "0:20",
      "end_time": "0:40",
      "period": "First Half",
      "priority": "CRITICAL/HIGH/MEDIUM",
      "reason": "First goal - detailed analysis needed",
      "analysis_focus": ["goal_buildup", "individual_skill", "defensive_error"],
      "score_at_start": {"home": 0, "away": 0},
      "score_at_end": {"home": 1, "away": 0},
      "key_players": ["#7 Player Name"],
      "context": "Opening minutes/Late pressure/Decisive moment"
    }
  ]
}

MANDATORY REQUIREMENTS:
- Create ONE focus_segment for EACH goal (minimum)
- Add focus_segments for: late-game pressure moments, controversial decisions, red cards, penalty situations
- Jersey patterns must be consistent for all players from same team
- Return ONLY valid JSON with NO trailing commas
- Use null for missing values, NOT empty strings
- Confidence level: "high" only if you see clear jersey number + name confirmation
- Mark unknown players as "Unknown Player #XX" with "low" confidence
- Ensure every goal has corresponding player in players_detected array"""

    def create_focus_prompt(self, skim_data: dict) -> str:
        focus_segments = skim_data.get('focus_segments', [])[:5]  # Analyze top 5 priority segments
        players = skim_data.get('players_detected', [])
        
        player_ref = "CONFIRMED PLAYERS IN VIDEO:\n"
        for p in players:
            conf = p.get('confidence_level', 'unknown')
            player_ref += f"  {p['player_id']} = {p['player_name']} ({p['team']}) [Confidence: {conf}]\n"
        
        seg_desc = "SEGMENTS TO ANALYZE IN DEPTH:\n"
        for i, seg in enumerate(focus_segments, 1):
            seg_desc += f"  {i}. {seg['start_time']} to {seg['end_time']}: {seg['reason']} (Priority: {seg['priority']})\n"
        
        goals_info = "\nGOALS SCORED IN MATCH:\n"
        for g in skim_data.get('goals_scored', []):
            goals_info += f"  Goal #{g['goal_number']} at {g['game_minute']}: {g['scorer_name']} ({g['scoring_team']})\n"
        
        pressure_info = "\nPRESSURE MOMENTS IDENTIFIED:\n"
        for pm in skim_data.get('pressure_moments', []):
            pressure_info += f"  {pm['game_minute']}: {pm['pressure_type']} (Intensity: {pm['intensity_rating']}/10)\n"
        
        return f"""PASS 2: DEEP PERFORMANCE & TACTICAL ANALYSIS

{player_ref}

{seg_desc}

{goals_info}

{pressure_info}

ANALYSIS REQUIREMENTS:

For each player visible in the focus segments, provide comprehensive analysis including:

1. TECHNICAL PERFORMANCE METRICS
2. TACTICAL CONTRIBUTION
3. PHYSICAL & MENTAL ATTRIBUTES
4. PRESSURE SITUATION PERFORMANCE
5. MATCH-CHANGING ACTIONS

OUTPUT STRUCTURE:
{{
  "analysis_metadata": {{
    "segments_analyzed": ["segment1_timeframe", "segment2_timeframe"],
    "total_analysis_duration_seconds": 0,
    "analysis_depth": "comprehensive",
    "key_focus_areas": ["goals", "pressure_moments", "tactical_shifts"]
  }},
  "players": [
    {{
      "player_id": "#7",
      "player_name": "Player Name",
      "team": "TEAM NAME",
      "position": "Forward",
      "analyzed_segments": [
        {{
          "timeframe": "0:20-0:40",
          "game_minute_range": "10'-15'",
          "minutes_analyzed": 0.33,
          
          "goals_and_assists": {{
            "goals_scored": 1,
            "assists_provided": 0,
            "goal_details": [
              {{
                "goal_number": 1,
                "timestamp": "0:36",
                "game_minute": "12'",
                "goal_type": "open_play",
                "buildup_involvement": "received_pass/made_run/dribbled_past_defender",
                "finish_quality": "clinical/good/lucky",
                "score_impact": {{"before": {{"home": 0, "away": 0}}, "after": {{"home": 1, "away": 0}}}},
                "pressure_situation": false,
                "difficulty_rating": 7
              }}
            ],
            "assist_details": []
          }},
          
          "ball_control_analysis": {{
            "total_touches": 5,
            "successful_touches": 5,
            "touch_quality": "excellent/good/average",
            "first_touch_rating": 8.5,
            "ball_control_time_seconds": 3,
            "times_dispossessed": 0,
            "heavy_touches": 0
          }},
          
          "passing_analysis": {{
            "total_passes": 2,
            "successful_passes": 2,
            "pass_accuracy_pct": 100.0,
            "key_passes": 0,
            "through_balls_attempted": 0,
            "through_balls_successful": 0,
            "crosses_attempted": 0,
            "crosses_successful": 0,
            "long_balls_attempted": 0,
            "long_balls_successful": 0,
            "pass_types": {{"short": 2, "medium": 0, "long": 0}},
            "progressive_passes": 0,
            "passes_under_pressure": 0
          }},
          
          "shooting_analysis": {{
            "total_shots": 1,
            "shots_on_target": 1,
            "shots_off_target": 0,
            "shots_blocked": 0,
            "goals_scored": 1,
            "expected_goals_xg": 0.65,
            "shot_accuracy_pct": 100.0,
            "shot_power_rating": 8,
            "shot_placement_rating": 9,
            "shots_from_inside_box": 1,
            "shots_from_outside_box": 0,
            "one_on_one_situations": 0,
            "shot_details": [
              {{
                "timestamp": "0:36",
                "body_part": "right_foot",
                "shot_type": "placed/power/chip/volley",
                "outcome": "goal",
                "xg_value": 0.65,
                "pressure_level": "low/medium/high"
              }}
            ]
          }},
          
          "dribbling_analysis": {{
            "dribbles_attempted": 2,
            "dribbles_successful": 2,
            "dribble_success_rate_pct": 100.0,
            "players_beaten": 1,
            "fouls_won_while_dribbling": 0,
            "possession_lost_while_dribbling": 0,
            "skillful_moves": ["step_over", "body_feint"],
            "1v1_duels_won": 1
          }},
          
          "defensive_contribution": {{
            "tackles_attempted": 0,
            "tackles_won": 0,
            "interceptions": 0,
            "clearances": 0,
            "blocks": 0,
            "aerial_duels_attempted": 0,
            "aerial_duels_won": 0,
            "recoveries": 0,
            "defensive_actions_in_own_box": 0,
            "tracking_back_instances": 0
          }},
          
          "physical_metrics": {{
            "distance_covered_meters": 60,
            "sprint_distance_meters": 20,
            "high_intensity_runs": 3,
            "sprints": 2,
            "top_speed_estimated_kmh": 32,
            "stamina_level": "high/medium/low",
            "work_rate_rating": 7
          }},
          
          "discipline_record": {{
            "fouls_committed": 0,
            "fouls_suffered": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "offsides": 0,
            "dissent_towards_officials": false
          }},
          
          "pressure_performance": {{
            "pressure_situations_faced": 1,
            "successful_under_pressure": 1,
            "pressure_composure_rating": 8,
            "late_game_performance": "N/A",
            "clutch_moments": ["Scored opening goal"],
            "decision_making_under_pressure": 8,
            "mental_strength_display": "confident/nervous/determined"
          }},
          
          "tactical_intelligence": {{
            "positioning_rating": 8,
            "off_ball_movement_quality": "excellent/good/average",
            "space_creation": 2,
            "defensive_positioning": 6,
            "tactical_awareness": 8,
            "runs_into_space": 3,
            "support_play_quality": "good",
            "team_play_rating": 7
          }},
          
          "match_impact_analysis": {{
            "overall_performance_rating": 8.5,
            "impact_on_scoreline": "decisive/significant/minimal",
            "game_changing_actions": ["Opening goal", "Created space for teammates"],
            "moments_of_brilliance": 1,
            "errors_leading_to_danger": 0,
            "leadership_displayed": false,
            "influence_on_team_morale": "positive/neutral/negative"
          }}
        }}
      ],
      
      "segment_aggregated_totals": {{
        "total_goals": 1,
        "total_assists": 0,
        "total_shots": 1,
        "total_key_passes": 0,
        "total_successful_dribbles": 2,
        "total_tackles_won": 0,
        "total_interceptions": 0,
        "overall_rating": 8.5,
        "performance_consistency": "excellent/good/inconsistent",
        "highlight_reel_moments": 1
      }},
      
      "contextual_performance": {{
        "performance_when_winning": "N/A",
        "performance_when_losing": "N/A",
        "performance_when_drawing": "8.5/10",
        "performance_in_pressure_moments": "8.0/10",
        "performance_vs_specific_opponent": {{"opponent_team": "TEAM NAME", "rating": 8.5}}
      }}
    }}
  ],
  
  "team_level_insights": [
    {{
      "team": "TEAM NAME",
      "segments_analyzed": ["0:20-0:40"],
      "tactical_setup": {{
        "formation": "4-3-3",
        "playing_style": "possession_based/counter_attacking/direct",
        "pressing_intensity": "high/medium/low",
        "defensive_line": "high/medium/deep"
      }},
      "collective_performance": {{
        "possession_estimated_pct": 55,
        "passing_accuracy_estimated_pct": 82,
        "attacking_efficiency": "high/medium/low",
        "defensive_solidity": "solid/vulnerable",
        "transition_speed": "fast/moderate/slow"
      }},
      "key_partnerships": [
        {{
          "players": ["#7 Player1", "#10 Player2"],
          "partnership_quality": "excellent/good/developing",
          "combinations": 3
        }}
      ]
    }}
  ],
  
  "match_narrative": {{
    "overall_match_quality": "excellent/good/average/poor",
    "entertainment_value": 8,
    "tactical_battle_rating": 7,
    "momentum_shifts": [
      {{
        "timestamp": "0:36",
        "description": "Home team takes lead",
        "impact": "significant"
      }}
    ],
    "turning_points": [
      {{
        "timestamp": "0:36",
        "event": "Opening goal",
        "significance": "Set tone for match"
      }}
    ]
  }}
}}

CRITICAL ANALYSIS RULES:
- Analyze ONLY players clearly visible in segments
- Use EXACT names from player reference list
- Be realistic with metrics - don't inflate numbers
- For pressure situations: rate composure, decision-making, execution
- For late-game scenarios (85'+ or stoppage time): emphasize mental strength
- For foul analysis: note if tactical, accidental, or dangerous
- Rate difficulty: 1-10 scale (10 = extremely difficult)
- All ratings on 1-10 scale unless specified otherwise
- NO trailing commas in JSON
- Use empty arrays [] not null for lists
- Keep response under 150KB
- If analyzing free kick: note technique, power, placement
- If analyzing penalty: note composure, placement, goalkeeper quality"""

    def fix_json(self, text: str) -> str:
        """Aggressively fix JSON issues."""
        # Remove markdown
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        # Extract JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        # Fix trailing commas before closing braces/brackets
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix multiple commas
        text = re.sub(r',+', ',', text)
        
        # Fix missing commas between objects
        text = re.sub(r'}\s*{', '},{', text)
        text = re.sub(r']\s*{', '],{', text)
        text = re.sub(r'}\s*\[', '},[', text)
        
        return text

    async def pass1_skim_analysis(self, video_file) -> dict:
        logger.info("=" * 60)
        logger.info("PASS 1: STRUCTURAL ANALYSIS")
        logger.info("=" * 60)
        
        for attempt in range(3):
            try:
                logger.info(f"Attempt {attempt + 1}/3...")
                
                response = self.client.models.generate_content(
                    model=self.skim_model_name,
                    contents=[video_file, self.create_skim_prompt()],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.05,
                        top_p=0.9,
                        response_mime_type="application/json"
                    )
                )
                
                if not response or not response.text:
                    logger.warning("Empty response, retrying...")
                    await asyncio.sleep(5)
                    continue

                cleaned = self.fix_json(response.text)
                result = json.loads(cleaned)
                
                # Validation checks
                if not result.get('focus_segments'):
                    logger.error("No focus_segments found - retrying")
                    await asyncio.sleep(5)
                    continue
                
                if not result.get('video_structure'):
                    logger.error("No video_structure found - retrying")
                    await asyncio.sleep(5)
                    continue
                
                # Log summary
                goals = len(result.get('goals_scored', []))
                players = len(result.get('players_detected', []))
                segments = len(result.get('focus_segments', []))
                fouls = len(result.get('fouls_and_cards', []))
                pressure = len(result.get('pressure_moments', []))
                
                logger.info(f"✓ Analysis complete:")
                logger.info(f"  - Goals: {goals}")
                logger.info(f"  - Players detected: {players}")
                logger.info(f"  - Focus segments: {segments}")
                logger.info(f"  - Fouls/Cards: {fouls}")
                logger.info(f"  - Pressure moments: {pressure}")
                
                # Validate player-team consistency
                self._validate_player_team_consistency(result)
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Position: {e.pos}, Message: {e.msg}")
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
            except Exception as e:
                logger.error(f"Pass 1 error: {type(e).__name__}: {e}")
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
        
        logger.error("❌ Pass 1 failed after 3 attempts")
        return {}

    def _validate_player_team_consistency(self, result: dict):
        """Validate that players are consistently assigned to teams based on jersey patterns."""
        team_home = result['video_structure'].get('team_home', '')
        team_away = result['video_structure'].get('team_away', '')
        
        home_jersey = result['video_structure'].get('team_home_jersey', {})
        away_jersey = result['video_structure'].get('team_away_jersey', {})
        
        logger.info("\nTeam Jersey Patterns:")
        logger.info(f"  {team_home}: {home_jersey.get('primary_colors', [])} {home_jersey.get('pattern', '')}")
        logger.info(f"  {team_away}: {away_jersey.get('primary_colors', [])} {away_jersey.get('pattern', '')}")
        
        # Check for any mismatches
        issues = 0
        for player in result.get('players_detected', []):
            player_jersey = player.get('jersey_pattern', {})
            player_team = player.get('team', '')
            
            # Validate jersey matches team
            if player_team == team_home:
                expected = home_jersey
            elif player_team == team_away:
                expected = away_jersey
            else:
                logger.warning(f"  ⚠️  {player['player_name']} assigned to unknown team: {player_team}")
                issues += 1
                continue
            
            # Basic color check
            player_colors = set(player_jersey.get('colors', []))
            expected_colors = set(expected.get('primary_colors', []))
            
            if not player_colors.intersection(expected_colors):
                logger.warning(f"  ⚠️  {player['player_name']} jersey colors {player_colors} "
                             f"don't match {player_team} colors {expected_colors}")
                issues += 1
        
        if issues == 0:
            logger.info("✓ All player-team assignments validated")
        else:
            logger.warning(f"⚠️  Found {issues} potential player-team assignment issues")

    async def pass2_focus_analysis(self, video_file, skim_data: dict) -> dict:
        logger.info("\n" + "=" * 60)
        logger.info("PASS 2: DEEP PERFORMANCE ANALYSIS")
        logger.info("=" * 60)
        
        if not skim_data.get('focus_segments'):
            logger.error("No focus segments available")
            return {}
        
        for attempt in range(3):
            try:
                logger.info(f"Attempt {attempt + 1}/3...")
                
                prompt = self.create_focus_prompt(skim_data)
                
                response = self.client.models.generate_content(
                    model=self.focus_model_name,
                    contents=[video_file, prompt],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        top_p=0.9,
                        response_mime_type="application/json",
                        max_output_tokens=16384
                    )
                )
                
                if not response or not response.text:
                    logger.warning("Empty response, retrying...")
                    await asyncio.sleep(5)
                    continue

                cleaned = self.fix_json(response.text)
                
                # Validate JSON structure
                if cleaned.count('{') != cleaned.count('}'):
                    logger.error(f"Unbalanced braces: {{ {cleaned.count('{')} vs }} {cleaned.count('}')}")
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                
                result = json.loads(cleaned)
                
                # Log summary
                players_analyzed = len(result.get('players', []))
                team_insights = len(result.get('team_level_insights', []))
                
                logger.info(f"✓ Analysis complete:")
                logger.info(f"  - Players analyzed: {players_analyzed}")
                logger.info(f"  - Team insights: {team_insights}")
                
                # Show top performers
                if result.get('players'):
                    top_players = sorted(
                        result['players'],
                        key=lambda p: p.get('segment_aggregated_totals', {}).get('overall_rating', 0),
                        reverse=True
                    )[:3]
                    
                    logger.info("\n  Top Performers:")
                    for i, player in enumerate(top_players, 1):
                        rating = player.get('segment_aggregated_totals', {}).get('overall_rating', 0)
                        logger.info(f"    {i}. {player['player_name']} - {rating}/10")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error at position {e.pos}: {e.msg}")
                logger.error(f"Context: ...{cleaned[max(0,e.pos-50):e.pos+50]}...")
                if attempt < 2:
                    await asyncio.sleep(15 * (attempt + 1))
            except Exception as e:
                logger.error(f"Pass 2 error: {type(e).__name__}: {e}")
                if attempt < 2:
                    await asyncio.sleep(15 * (attempt + 1))
        
        logger.error("❌ Pass 2 failed after 3 attempts")
        return {}

    def merge_and_save(self, skim_data: dict, focus_data: dict, base_name: str):
        """Merge both passes and save comprehensive analysis."""
        
        merged = {
            'match_information': {
                'video_structure': skim_data.get('video_structure', {}),
                'match_events': {
                    'goals_scored': skim_data.get('goals_scored', []),
                    'fouls_and_cards': skim_data.get('fouls_and_cards', []),
                    'set_pieces': skim_data.get('set_pieces', []),
                    'substitutions': skim_data.get('substitutions', []),
                    'key_moments': skim_data.get('key_moments', []),
                    'pressure_moments': skim_data.get('pressure_moments', [])
                },
                'tactical_observations': skim_data.get('tactical_observations', []),
                'focus_segments_analyzed': skim_data.get('focus_segments', []),
                'analysis_metadata': focus_data.get('analysis_metadata', {})
            },
            'player_performances': [],
            'team_analysis': focus_data.get('team_level_insights', []),
            'match_narrative': focus_data.get('match_narrative', {}),
            'statistical_summary': {}
        }
        
        # Merge player data
        skim_players = {p['player_id']: p for p in skim_data.get('players_detected', [])}
        focus_players = {p['player_id']: p for p in focus_data.get('players', [])}
        
        all_player_ids = set(skim_players.keys()) | set(focus_players.keys())
        
        for pid in all_player_ids:
            skim_player = skim_players.get(pid, {})
            focus_player = focus_players.get(pid, {})
            
            merged_player = {
                'player_id': pid,
                'player_name': skim_player.get('player_name') or focus_player.get('player_name', 'Unknown'),
                'team': skim_player.get('team') or focus_player.get('team', 'Unknown'),
                'position': skim_player.get('position') or focus_player.get('position', 'Unknown'),
                'confidence_level': skim_player.get('confidence_level', 'unknown'),
                'detection_summary': {
                    'jersey_pattern': skim_player.get('jersey_pattern', {}),
                    'visible_periods': skim_player.get('visible_periods', []),
                    'is_starter': skim_player.get('is_starter', False),
                    'estimated_minutes': skim_player.get('estimated_minutes', 0),
                    'quick_stats': {
                        'goals_detected': skim_player.get('goals_scored_count', 0),
                        'assists_detected': skim_player.get('assists_count', 0)
                    }
                },
                'detailed_performance_analysis': focus_player.get('analyzed_segments', []),
                'aggregated_statistics': focus_player.get('segment_aggregated_totals', {}),
                'contextual_performance': focus_player.get('contextual_performance', {})
            }
            
            merged['player_performances'].append(merged_player)
        
        # Sort by overall rating
        merged['player_performances'].sort(
            key=lambda p: p.get('aggregated_statistics', {}).get('overall_rating', 0),
            reverse=True
        )
        
        # Calculate statistical summary
        merged['statistical_summary'] = self._calculate_summary_stats(merged)
        
        # Save outputs
        os.makedirs("outputfoot", exist_ok=True)
        
        # Complete JSON
        json_path = f"outputfoot/{base_name}_complete_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved complete analysis: {json_path}")
        
        # Player summary CSV
        self._save_player_csv(merged, base_name)
        
        # Match events CSV
        self._save_events_csv(merged, base_name)
        
        # Performance report
        self._save_performance_report(merged, base_name)
        
        return merged

    def _calculate_summary_stats(self, merged: dict) -> dict:
        """Calculate aggregate statistics for the match."""
        goals = merged['match_information']['match_events']['goals_scored']
        fouls = merged['match_information']['match_events']['fouls_and_cards']
        players = merged['player_performances']
        
        teams = set()
        for p in players:
            if p['team'] != 'Unknown':
                teams.add(p['team'])
        
        summary = {
            'total_goals': len(goals),
            'total_fouls': len(fouls),
            'total_players_detected': len(players),
            'total_players_analyzed': sum(1 for p in players if p['detailed_performance_analysis']),
            'yellow_cards': sum(1 for f in fouls if f.get('card_given') == 'yellow'),
            'red_cards': sum(1 for f in fouls if f.get('card_given') == 'red'),
            'pressure_moments_count': len(merged['match_information']['match_events']['pressure_moments']),
            'set_pieces_count': len(merged['match_information']['match_events']['set_pieces']),
            'substitutions_count': len(merged['match_information']['match_events']['substitutions']),
            'teams': list(teams)
        }
        
        # Team-specific stats
        for team in teams:
            team_goals = sum(1 for g in goals if g.get('scoring_team') == team)
            team_fouls = sum(1 for f in fouls if f.get('foul_by_team') == team)
            team_players = [p for p in players if p['team'] == team]
            
            summary[f'{team}_stats'] = {
                'goals': team_goals,
                'fouls': team_fouls,
                'players_detected': len(team_players),
                'top_performer': team_players[0]['player_name'] if team_players else 'N/A',
                'top_rating': team_players[0].get('aggregated_statistics', {}).get('overall_rating', 0) if team_players else 0
            }
        
        return summary

    def _save_player_csv(self, merged: dict, base_name: str):
        """Save player statistics as CSV."""
        rows = []
        
        for player in merged['player_performances']:
            agg_stats = player.get('aggregated_statistics', {})
            detection = player.get('detection_summary', {})
            
            row = {
                'player_id': player['player_id'],
                'player_name': player['player_name'],
                'team': player['team'],
                'position': player['position'],
                'confidence': player.get('confidence_level', 'unknown'),
                'minutes_estimated': detection.get('estimated_minutes', 0),
                'goals': agg_stats.get('total_goals', 0),
                'assists': agg_stats.get('total_assists', 0),
                'shots': agg_stats.get('total_shots', 0),
                'key_passes': agg_stats.get('total_key_passes', 0),
                'successful_dribbles': agg_stats.get('total_successful_dribbles', 0),
                'tackles_won': agg_stats.get('total_tackles_won', 0),
                'interceptions': agg_stats.get('total_interceptions', 0),
                'overall_rating': agg_stats.get('overall_rating', 0),
                'performance_consistency': agg_stats.get('performance_consistency', 'N/A'),
                'highlight_moments': agg_stats.get('highlight_reel_moments', 0)
            }
            
            # Add contextual performance
            context = player.get('contextual_performance', {})
            row['performance_in_pressure'] = context.get('performance_in_pressure_moments', 'N/A')
            row['performance_when_drawing'] = context.get('performance_when_drawing', 'N/A')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = f"outputfoot/{base_name}_player_statistics.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved player statistics: {csv_path}")

    def _save_events_csv(self, merged: dict, base_name: str):
        """Save match events as CSV."""
        events = []
        
        # Goals
        for goal in merged['match_information']['match_events']['goals_scored']:
            events.append({
                'timestamp': goal.get('timestamp'),
                'game_minute': goal.get('game_minute'),
                'event_type': 'GOAL',
                'team': goal.get('scoring_team'),
                'player': goal.get('scorer_name'),
                'details': f"{goal.get('goal_type')} - {goal.get('body_part')}",
                'score_after': f"{goal['score_after'].get('home', 0)}-{goal['score_after'].get('away', 0)}"
            })
        
        # Fouls and cards
        for foul in merged['match_information']['match_events']['fouls_and_cards']:
            card = foul.get('card_given', 'none')
            events.append({
                'timestamp': foul.get('timestamp'),
                'game_minute': foul.get('game_minute'),
                'event_type': f"FOUL ({card.upper()})" if card != 'none' else 'FOUL',
                'team': foul.get('foul_by_team'),
                'player': foul.get('foul_by_name'),
                'details': f"{foul.get('foul_type')} foul on {foul.get('foul_on_name')}",
                'score_after': ''
            })
        
        # Substitutions
        for sub in merged['match_information']['match_events']['substitutions']:
            events.append({
                'timestamp': sub.get('timestamp'),
                'game_minute': sub.get('game_minute'),
                'event_type': 'SUBSTITUTION',
                'team': sub.get('team'),
                'player': f"{sub.get('player_off_name')} OFF, {sub.get('player_on_name')} ON",
                'details': sub.get('reason_visible', ''),
                'score_after': ''
            })
        
        # Pressure moments
        for pressure in merged['match_information']['match_events']['pressure_moments']:
            events.append({
                'timestamp': pressure.get('timestamp'),
                'game_minute': pressure.get('game_minute'),
                'event_type': 'PRESSURE MOMENT',
                'team': pressure.get('team_under_pressure'),
                'player': '',
                'details': f"{pressure.get('pressure_type')} (Intensity: {pressure.get('intensity_rating')}/10)",
                'score_after': ''
            })
        
        if events:
            df = pd.DataFrame(events)
            df = df.sort_values('timestamp')
            csv_path = f"outputfoot/{base_name}_match_events.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved match events: {csv_path}")

    def _save_performance_report(self, merged: dict, base_name: str):
        """Save human-readable performance report."""
        report_path = f"outputfoot/{base_name}_performance_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FOOTBALL MATCH ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Match info
            video_struct = merged['match_information']['video_structure']
            f.write(f"Match: {video_struct.get('team_home', 'Unknown')} vs {video_struct.get('team_away', 'Unknown')}\n")
            f.write(f"Competition: {video_struct.get('competition', 'Unknown')}\n")
            f.write(f"Final Score: {video_struct['final_score'].get('home', 0)} - {video_struct['final_score'].get('away', 0)}\n\n")
            
            # Summary stats
            summary = merged['statistical_summary']
            f.write("MATCH STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Goals: {summary['total_goals']}\n")
            f.write(f"Total Fouls: {summary['total_fouls']}\n")
            f.write(f"Yellow Cards: {summary['yellow_cards']}\n")
            f.write(f"Red Cards: {summary['red_cards']}\n")
            f.write(f"Pressure Moments: {summary['pressure_moments_count']}\n")
            f.write(f"Set Pieces: {summary['set_pieces_count']}\n")
            f.write(f"Substitutions: {summary['substitutions_count']}\n\n")
            
            # Top performers
            f.write("TOP PERFORMERS\n")
            f.write("-" * 80 + "\n")
            for i, player in enumerate(merged['player_performances'][:5], 1):
                rating = player.get('aggregated_statistics', {}).get('overall_rating', 0)
                goals = player.get('aggregated_statistics', {}).get('total_goals', 0)
                assists = player.get('aggregated_statistics', {}).get('total_assists', 0)
                
                f.write(f"{i}. {player['player_name']} ({player['team']}) - Rating: {rating}/10\n")
                f.write(f"   Position: {player['position']} | Goals: {goals} | Assists: {assists}\n")
                
                if player.get('detailed_performance_analysis'):
                    segments = player['detailed_performance_analysis']
                    if segments:
                        impact = segments[0].get('match_impact_analysis', {})
                        actions = impact.get('game_changing_actions', [])
                        if actions:
                            f.write(f"   Key Actions: {', '.join(actions)}\n")
                f.write("\n")
            
            # Goals chronology
            f.write("GOALS CHRONOLOGY\n")
            f.write("-" * 80 + "\n")
            for goal in merged['match_information']['match_events']['goals_scored']:
                f.write(f"{goal['game_minute']}: {goal['scorer_name']} ({goal['scoring_team']})\n")
                f.write(f"   Type: {goal['goal_type']} | Body part: {goal.get('body_part', 'unknown')}\n")
                if goal.get('assist_by_name'):
                    f.write(f"   Assist: {goal['assist_by_name']}\n")
                f.write(f"   Score: {goal['score_after']['home']}-{goal['score_after']['away']}\n\n")
            
            # Pressure moments
            if merged['match_information']['match_events']['pressure_moments']:
                f.write("PRESSURE MOMENTS\n")
                f.write("-" * 80 + "\n")
                for pm in merged['match_information']['match_events']['pressure_moments']:
                    f.write(f"{pm['game_minute']}: {pm['pressure_type']}\n")
                    f.write(f"   Team under pressure: {pm['team_under_pressure']}\n")
                    f.write(f"   Intensity: {pm['intensity_rating']}/10\n")
                    f.write(f"   Outcome: {pm.get('outcome', 'Unknown')}\n\n")
            
            # Team insights
            if merged.get('team_analysis'):
                f.write("TEAM TACTICAL ANALYSIS\n")
                f.write("-" * 80 + "\n")
                for team_insight in merged['team_analysis']:
                    f.write(f"Team: {team_insight['team']}\n")
                    tactical = team_insight.get('tactical_setup', {})
                    f.write(f"   Formation: {tactical.get('formation', 'Unknown')}\n")
                    f.write(f"   Playing Style: {tactical.get('playing_style', 'Unknown')}\n")
                    f.write(f"   Pressing Intensity: {tactical.get('pressing_intensity', 'Unknown')}\n")
                    
                    collective = team_insight.get('collective_performance', {})
                    if collective:
                        f.write(f"   Possession: ~{collective.get('possession_estimated_pct', 0)}%\n")
                        f.write(f"   Passing Accuracy: ~{collective.get('passing_accuracy_estimated_pct', 0)}%\n")
                    f.write("\n")
            
            # Match narrative
            narrative = merged.get('match_narrative', {})
            if narrative:
                f.write("MATCH NARRATIVE\n")
                f.write("-" * 80 + "\n")
                f.write(f"Match Quality: {narrative.get('overall_match_quality', 'Unknown')}\n")
                f.write(f"Entertainment Value: {narrative.get('entertainment_value', 0)}/10\n")
                f.write(f"Tactical Battle: {narrative.get('tactical_battle_rating', 0)}/10\n\n")
                
                if narrative.get('turning_points'):
                    f.write("Key Turning Points:\n")
                    for tp in narrative['turning_points']:
                        f.write(f"   {tp['timestamp']}: {tp['event']} - {tp.get('significance', '')}\n")
        
        logger.info(f"✓ Saved performance report: {report_path}")

    async def run(self, video_path: str):
        """Main execution flow."""
        logger.info("=" * 60)
        logger.info("TWO-PASS FOOTBALL ANALYZER")
        logger.info("=" * 60)
        logger.info(f"Video: {video_path}\n")
        
        video_file = None
        try:
            # Upload video
            logger.info("📤 Uploading video...")
            video_file = self.client.files.upload(file=video_path)
            
            # Wait for processing
            wait_time = 0
            max_wait = 300  # 5 minutes
            while video_file.state.name == "PROCESSING" and wait_time < max_wait:
                await asyncio.sleep(5)
                wait_time += 5
                video_file = self.client.files.get(name=video_file.name)
                if wait_time % 30 == 0:
                    logger.info(f"   Still processing... ({wait_time}s)")
            
            if video_file.state.name != "ACTIVE":
                logger.error(f"❌ Upload failed: {video_file.state.name}")
                return
            
            logger.info("✓ Upload complete\n")
            
            # PASS 1: Structural Analysis
            skim_data = await self.pass1_skim_analysis(video_file)
            if not skim_data or not skim_data.get('focus_segments'):
                logger.error("❌ Pass 1 failed - cannot proceed")
                return
            
            # PASS 2: Deep Analysis
            focus_data = await self.pass2_focus_analysis(video_file, skim_data)
            if not focus_data:
                logger.warning("⚠️  Pass 2 failed - saving Pass 1 data only")
                focus_data = {
                    'players': [],
                    'team_level_insights': [],
                    'match_narrative': {},
                    'analysis_metadata': {'status': 'Pass 2 failed'}
                }
            
            # Merge and save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            team_home = skim_data['video_structure'].get('team_home', 'Team1').replace(' ', '_')
            team_away = skim_data['video_structure'].get('team_away', 'Team2').replace(' ', '_')
            base_name = f"{team_home}_vs_{team_away}_{timestamp}"
            
            merged = self.merge_and_save(skim_data, focus_data, base_name)
            
            # Final summary
            logger.info("\n" + "=" * 60)
            logger.info("✅ ANALYSIS COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Match: {team_home} vs {team_away}")
            logger.info(f"Final Score: {skim_data['video_structure']['final_score']}")
            logger.info(f"Players Detected: {merged['statistical_summary']['total_players_detected']}")
            logger.info(f"Players Analyzed: {merged['statistical_summary']['total_players_analyzed']}")
            logger.info(f"Goals: {merged['statistical_summary']['total_goals']}")
            logger.info(f"Fouls: {merged['statistical_summary']['total_fouls']}")
            logger.info(f"Pressure Moments: {merged['statistical_summary']['pressure_moments_count']}")
            logger.info(f"\nOutput files saved in 'outputfoot/' directory")
            
        except Exception as e:
            logger.critical(f"💥 Critical error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            # Cleanup
            if video_file:
                try:
                    self.client.files.delete(name=video_file.name)
                    logger.info("✓ Video file cleanup complete")
                except Exception as e:
                    logger.warning(f"Cleanup warning: {e}")


async def main():
    """Entry point."""
    import sys
    
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    # Get video path
    if len(sys.argv) < 2:
        print("Usage: python TwoPassFootballAnalyzer.py <video_path>")
        print("Example: python TwoPassFootballAnalyzer.py match.mp4")
        return
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Run analyzer
    analyzer = TwoPassFootballAnalyzer(api_key)
    await analyzer.run(video_path)


if __name__ == "__main__":
    asyncio.run(main())