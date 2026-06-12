from google import genai
import pandas as pd
import json
from datetime import datetime
import os
import logging
import asyncio
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONSTANT SCHEMA DEFINITION
MATCH_SCHEMA = {
    "schema_version": "1.0.0",
    "match_metadata": {
        "match_id": "",
        "competition": "",
        "venue": "",
        "date": "",
        "duration_seconds": 0
    },
    "teams": {
        "home": {
            "name": "",
            "score": 0,
            "jersey": {"colors": [], "pattern": "", "shorts": ""}
        },
        "away": {
            "name": "",
            "score": 0,
            "jersey": {"colors": [], "pattern": "", "shorts": ""}
        }
    },
    "players": [],
    "events": {
        "goals": [],
        "fouls": [],
        "cards": [],
        "substitutions": [],
        "pressure_moments": []
    },
    "statistics": {
        "team_home": {},
        "team_away": {},
        "match_totals": {}
    },
    "analysis_metadata": {
        "analyzed_at": "",
        "segments_analyzed": [],
        "analysis_confidence": ""
    }
}

PLAYER_SCHEMA = {
    "player_id": "",
    "jersey_number": 0,
    "name": "",
    "team": "",
    "position": "",
    "starter": False,
    "minutes_played": 0,
    "validation": {
        "confidence": "",
        "verification_source": "",
        "jersey_confirmed": False,
        "name_confirmed": False
    },
    "performance": {
        "goals": 0,
        "assists": 0,
        "shots": 0,
        "shots_on_target": 0,
        "passes": 0,
        "passes_completed": 0,
        "pass_accuracy": 0.0,
        "key_passes": 0,
        "dribbles": 0,
        "dribbles_successful": 0,
        "tackles": 0,
        "tackles_won": 0,
        "interceptions": 0,
        "fouls_committed": 0,
        "fouls_won": 0,
        "rating": 0.0
    },
    "pressure_performance": {
        "situations_faced": 0,
        "composure_rating": 0.0,
        "successful_actions": 0,
        "errors_under_pressure": 0,
        "clutch_moments": []
    }
}

EVENT_SCHEMA = {
    "goal": {
        "event_id": "",
        "timestamp": "",
        "minute": "",
        "player_id": "",
        "player_name": "",
        "team": "",
        "assist_player_id": None,
        "assist_player_name": None,
        "type": "",
        "body_part": "",
        "score_before": {"home": 0, "away": 0},
        "score_after": {"home": 0, "away": 0}
    },
    "foul": {
        "event_id": "",
        "timestamp": "",
        "minute": "",
        "player_id": "",
        "player_name": "",
        "team": "",
        "type": "",
        "card": None
    },
    "substitution": {
        "event_id": "",
        "timestamp": "",
        "minute": "",
        "team": "",
        "player_off_id": "",
        "player_off_name": "",
        "player_on_id": "",
        "player_on_name": ""
    },
    "pressure_moment": {
        "event_id": "",
        "timestamp": "",
        "minute": "",
        "type": "",
        "team_under_pressure": "",
        "intensity": 0,
        "score_situation": {"home": 0, "away": 0},
        "outcome": "",
        "key_players": []
    }
}

class TwoPassFootballAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.skim_model_name = 'models/gemini-2.5-flash'
        self.focus_model_name = 'models/gemini-2.5-flash'
    
    def create_skim_prompt(self) -> str:
        return """PASS 1: MATCH STRUCTURE & PLAYER IDENTIFICATION

EXTRACT INTO THIS EXACT STRUCTURE:

{
  "match_metadata": {
    "competition": "string",
    "venue": "string or null",
    "date": "YYYY-MM-DD or null",
    "duration_seconds": number
  },
  "teams": {
    "home": {
      "name": "string",
      "score": number,
      "jersey": {"colors": ["string"], "pattern": "string", "shorts": "string"}
    },
    "away": {
      "name": "string", 
      "score": number,
      "jersey": {"colors": ["string"], "pattern": "string", "shorts": "string"}
    }
  },
  "players_detected": [
    {
      "jersey_number": number,
      "name": "string OR null if unknown",
      "team": "home/away",
      "position": "GK/DEF/MID/FWD",
      "starter": boolean,
      "minutes_played": number,
      "verification": {
        "confidence": "high/medium/low",
        "source": "replay/graphic/commentary",
        "jersey_visible": boolean,
        "name_visible": boolean
      }
    }
  ],
  "events": {
    "goals": [
      {
        "timestamp": "MM:SS",
        "minute": "number'",
        "jersey_number": number,
        "player_name": "string or null",
        "team": "home/away",
        "assist_jersey": number or null,
        "assist_name": "string or null",
        "type": "open_play/penalty/free_kick/header/counter",
        "body_part": "right_foot/left_foot/header",
        "score_after": {"home": number, "away": number}
      }
    ],
    "fouls": [
      {
        "timestamp": "MM:SS",
        "minute": "number'",
        "jersey_number": number,
        "player_name": "string or null",
        "team": "home/away",
        "type": "tactical/dangerous/normal",
        "card": "yellow/red/none"
      }
    ],
    "substitutions": [
      {
        "timestamp": "MM:SS",
        "minute": "number'",
        "team": "home/away",
        "off_jersey": number,
        "off_name": "string or null",
        "on_jersey": number,
        "on_name": "string or null"
      }
    ],
    "pressure_moments": [
      {
        "timestamp": "MM:SS",
        "minute": "number'",
        "type": "late_game_drama/penalty_situation/red_card_impact/injury_time/close_score/last_minute_attack",
        "team_under_pressure": "home/away",
        "intensity": number (1-10),
        "score_situation": {"home": number, "away": number},
        "outcome": "goal_scored/defended_successfully/opportunity_missed/ongoing",
        "key_players": [number]
      }
    ]
  },
  "focus_segments": [
    {
      "start": "MM:SS",
      "end": "MM:SS",
      "priority": "critical/high/medium",
      "reason": "string",
      "players": [number]
    }
  ]
}

PRESSURE MOMENT DETECTION:
✓ Late game (85'+ or injury time) when score is close (0-1 goal difference)
✓ Penalty situations
✓ Red card impacts (team down to 10 players)
✓ Last-minute attacks when losing/drawing
✓ Goalkeeper saves in critical moments
✓ Defensive stands under sustained attack

VALIDATION RULES:
✓ Use jersey_number (integer) consistently as primary ID
✓ team must be "home" or "away" (not team names)
✓ If name unknown, use null not "Unknown"
✓ All numbers must be actual numbers, not strings
✓ Include focus_segment for EACH goal + pressure moment
✓ Intensity: 1-3 (low), 4-6 (medium), 7-8 (high), 9-10 (critical)"""

    def create_focus_prompt(self, skim_data: dict) -> str:
        segments = skim_data.get('focus_segments', [])[:5]
        players = skim_data.get('players_detected', [])
        pressure_moments = skim_data.get('events', {}).get('pressure_moments', [])
        
        player_ref = "VERIFIED PLAYERS:\n"
        for p in players:
            num = p.get('jersey_number')
            name = p.get('name') or f"#{num}"
            team = p.get('team')
            conf = p.get('verification', {}).get('confidence', 'unknown')
            player_ref += f"  #{num} = {name} ({team}) [{conf}]\n"
        
        seg_desc = "\nANALYZE SEGMENTS:\n"
        for i, seg in enumerate(segments, 1):
            seg_desc += f"  {i}. {seg['start']}-{seg['end']}: {seg['reason']}\n"
        
        pressure_desc = "\nPRESSURE MOMENTS DETECTED:\n"
        if pressure_moments:
            for pm in pressure_moments:
                pressure_desc += f"  {pm['minute']}: {pm['type']} (Intensity: {pm['intensity']}/10)\n"
        else:
            pressure_desc += "  None detected\n"
        
        return f"""PASS 2: PLAYER PERFORMANCE ANALYSIS

{player_ref}
{seg_desc}
{pressure_desc}

OUTPUT THIS EXACT STRUCTURE:

{{
  "players_analyzed": [
    {{
      "jersey_number": number,
      "performance": {{
        "goals": number,
        "assists": number,
        "shots": number,
        "shots_on_target": number,
        "passes": number,
        "passes_completed": number,
        "pass_accuracy": float,
        "key_passes": number,
        "dribbles": number,
        "dribbles_successful": number,
        "tackles": number,
        "tackles_won": number,
        "interceptions": number,
        "fouls_committed": number,
        "fouls_won": number,
        "rating": float
      }},
      "pressure_performance": {{
        "situations_faced": number,
        "composure_rating": float (1-10),
        "successful_actions": number,
        "errors_under_pressure": number,
        "clutch_moments": ["description"]
      }}
    }}
  ],
  "team_statistics": {{
    "home": {{
      "possession_pct": float,
      "pass_accuracy_pct": float,
      "shots": number,
      "shots_on_target": number
    }},
    "away": {{
      "possession_pct": float,
      "pass_accuracy_pct": float,
      "shots": number,
      "shots_on_target": number
    }}
  }}
}}

PRESSURE ANALYSIS INSTRUCTIONS:
- For players visible in pressure moments, analyze their composure and decision-making
- Rate composure: 1-10 (1=panicked, 5=average, 10=ice-cold)
- Count successful actions (passes completed, tackles won, saves made under pressure)
- Note errors (misplaced passes, fouls, missed chances under pressure)
- Identify clutch moments (game-saving tackles, pressure goals, critical saves)

RULES:
- Use jersey_number (integer) to match Pass 1
- All numeric fields must be numbers (not strings)
- Ratings: 0.0-10.0 scale
- Percentages: 0.0-100.0
- Only analyze players visible in segments"""

    def fix_json(self, text: str) -> str:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        text = re.sub(r',+', ',', text)
        
        return text

    def normalize_to_schema(self, skim_data: dict, focus_data: dict, video_path: str) -> dict:
        """Normalize extracted data into constant schema."""
        output = json.loads(json.dumps(MATCH_SCHEMA))  # Deep copy
        
        # Match metadata
        match_meta = skim_data.get('match_metadata', {})
        teams = skim_data.get('teams', {})
        home_team = teams.get('home', {}).get('name', 'Team Home')
        away_team = teams.get('away', {}).get('name', 'Team Away')
        
        output['match_metadata'] = {
            'match_id': f"{home_team}_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'competition': match_meta.get('competition', 'Unknown'),
            'venue': match_meta.get('venue'),
            'date': match_meta.get('date'),
            'duration_seconds': match_meta.get('duration_seconds', 0)
        }
        
        # Teams
        output['teams']['home'] = teams.get('home', {
            'name': 'Team Home',
            'score': 0,
            'jersey': {'colors': [], 'pattern': '', 'shorts': ''}
        })
        output['teams']['away'] = teams.get('away', {
            'name': 'Team Away', 
            'score': 0,
            'jersey': {'colors': [], 'pattern': '', 'shorts': ''}
        })
        
        # Build player lookup
        players_detected = skim_data.get('players_detected', [])
        focus_players = {p.get('jersey_number'): p.get('performance', {}) 
                        for p in focus_data.get('players_analyzed', [])}
        
        # Normalize players
        for player_data in players_detected:
            player = json.loads(json.dumps(PLAYER_SCHEMA))
            
            jersey_num = player_data.get('jersey_number')
            player['player_id'] = f"player_{jersey_num}"
            player['jersey_number'] = jersey_num
            player['name'] = player_data.get('name') or f"Player #{jersey_num}"
            
            # Convert team from home/away to actual name
            team_type = player_data.get('team', 'home')
            player['team'] = output['teams'][team_type]['name']
            
            player['position'] = player_data.get('position', 'Unknown')
            player['starter'] = player_data.get('starter', False)
            player['minutes_played'] = player_data.get('minutes_played', 0)
            
            # Validation
            verification = player_data.get('verification', {})
            player['validation'] = {
                'confidence': verification.get('confidence', 'low'),
                'verification_source': verification.get('source', 'unknown'),
                'jersey_confirmed': verification.get('jersey_visible', False),
                'name_confirmed': verification.get('name_visible', False)
            }
            
            # Performance (from Pass 2)
            if jersey_num in focus_players:
                perf = focus_players[jersey_num]
                player['performance'] = {
                    'goals': perf.get('goals', 0),
                    'assists': perf.get('assists', 0),
                    'shots': perf.get('shots', 0),
                    'shots_on_target': perf.get('shots_on_target', 0),
                    'passes': perf.get('passes', 0),
                    'passes_completed': perf.get('passes_completed', 0),
                    'pass_accuracy': perf.get('pass_accuracy', 0.0),
                    'key_passes': perf.get('key_passes', 0),
                    'dribbles': perf.get('dribbles', 0),
                    'dribbles_successful': perf.get('dribbles_successful', 0),
                    'tackles': perf.get('tackles', 0),
                    'tackles_won': perf.get('tackles_won', 0),
                    'interceptions': perf.get('interceptions', 0),
                    'fouls_committed': perf.get('fouls_committed', 0),
                    'fouls_won': perf.get('fouls_won', 0),
                    'rating': perf.get('rating', 0.0)
                }
                
                # Pressure performance
                pressure_perf = perf.get('pressure_performance', {})
                player['pressure_performance'] = {
                    'situations_faced': pressure_perf.get('situations_faced', 0),
                    'composure_rating': pressure_perf.get('composure_rating', 0.0),
                    'successful_actions': pressure_perf.get('successful_actions', 0),
                    'errors_under_pressure': pressure_perf.get('errors_under_pressure', 0),
                    'clutch_moments': pressure_perf.get('clutch_moments', [])
                }
            
            output['players'].append(player)
        
        # Normalize events
        events = skim_data.get('events', {})
        
        # Goals
        for goal in events.get('goals', []):
            goal_event = json.loads(json.dumps(EVENT_SCHEMA['goal']))
            goal_event['event_id'] = f"goal_{len(output['events']['goals']) + 1}"
            goal_event['timestamp'] = goal.get('timestamp', '')
            goal_event['minute'] = goal.get('minute', '')
            
            jersey_num = goal.get('jersey_number')
            goal_event['player_id'] = f"player_{jersey_num}"
            goal_event['player_name'] = goal.get('player_name') or f"Player #{jersey_num}"
            
            team_type = goal.get('team', 'home')
            goal_event['team'] = output['teams'][team_type]['name']
            
            assist_jersey = goal.get('assist_jersey')
            if assist_jersey:
                goal_event['assist_player_id'] = f"player_{assist_jersey}"
                goal_event['assist_player_name'] = goal.get('assist_name')
            
            goal_event['type'] = goal.get('type', 'open_play')
            goal_event['body_part'] = goal.get('body_part', 'unknown')
            goal_event['score_before'] = goal.get('score_before', {'home': 0, 'away': 0})
            goal_event['score_after'] = goal.get('score_after', {'home': 0, 'away': 0})
            
            output['events']['goals'].append(goal_event)
        
        # Fouls
        for foul in events.get('fouls', []):
            foul_event = json.loads(json.dumps(EVENT_SCHEMA['foul']))
            foul_event['event_id'] = f"foul_{len(output['events']['fouls']) + 1}"
            foul_event['timestamp'] = foul.get('timestamp', '')
            foul_event['minute'] = foul.get('minute', '')
            
            jersey_num = foul.get('jersey_number')
            foul_event['player_id'] = f"player_{jersey_num}"
            foul_event['player_name'] = foul.get('player_name') or f"Player #{jersey_num}"
            
            team_type = foul.get('team', 'home')
            foul_event['team'] = output['teams'][team_type]['name']
            
            foul_event['type'] = foul.get('type', 'normal')
            
            card = foul.get('card', 'none')
            if card and card != 'none':
                foul_event['card'] = card
                output['events']['cards'].append({
                    'event_id': f"card_{len(output['events']['cards']) + 1}",
                    'timestamp': foul_event['timestamp'],
                    'minute': foul_event['minute'],
                    'player_id': foul_event['player_id'],
                    'player_name': foul_event['player_name'],
                    'team': foul_event['team'],
                    'card_type': card
                })
            
            output['events']['fouls'].append(foul_event)
        
        # Substitutions
        for sub in events.get('substitutions', []):
            sub_event = json.loads(json.dumps(EVENT_SCHEMA['substitution']))
            sub_event['event_id'] = f"sub_{len(output['events']['substitutions']) + 1}"
            sub_event['timestamp'] = sub.get('timestamp', '')
            sub_event['minute'] = sub.get('minute', '')
            
            team_type = sub.get('team', 'home')
            sub_event['team'] = output['teams'][team_type]['name']
            
            off_jersey = sub.get('off_jersey')
            on_jersey = sub.get('on_jersey')
            
            sub_event['player_off_id'] = f"player_{off_jersey}"
            sub_event['player_off_name'] = sub.get('off_name') or f"Player #{off_jersey}"
            sub_event['player_on_id'] = f"player_{on_jersey}"
            sub_event['player_on_name'] = sub.get('on_name') or f"Player #{on_jersey}"
            
            output['events']['substitutions'].append(sub_event)
        
        # Pressure Moments
        for pressure in events.get('pressure_moments', []):
            pressure_event = json.loads(json.dumps(EVENT_SCHEMA['pressure_moment']))
            pressure_event['event_id'] = f"pressure_{len(output['events']['pressure_moments']) + 1}"
            pressure_event['timestamp'] = pressure.get('timestamp', '')
            pressure_event['minute'] = pressure.get('minute', '')
            pressure_event['type'] = pressure.get('type', 'unknown')
            
            team_type = pressure.get('team_under_pressure', 'home')
            pressure_event['team_under_pressure'] = output['teams'][team_type]['name']
            
            pressure_event['intensity'] = pressure.get('intensity', 0)
            pressure_event['score_situation'] = pressure.get('score_situation', {'home': 0, 'away': 0})
            pressure_event['outcome'] = pressure.get('outcome', 'unknown')
            
            # Map key players
            key_players = []
            for jersey_num in pressure.get('key_players', []):
                key_players.append({
                    'player_id': f"player_{jersey_num}",
                    'jersey_number': jersey_num
                })
            pressure_event['key_players'] = key_players
            
            output['events']['pressure_moments'].append(pressure_event)
        
        # Statistics
        team_stats = focus_data.get('team_statistics', {})
        output['statistics']['team_home'] = team_stats.get('home', {
            'possession_pct': 0.0,
            'pass_accuracy_pct': 0.0,
            'shots': 0,
            'shots_on_target': 0
        })
        output['statistics']['team_away'] = team_stats.get('away', {
            'possession_pct': 0.0,
            'pass_accuracy_pct': 0.0,
            'shots': 0,
            'shots_on_target': 0
        })
        
        # Match totals
        output['statistics']['match_totals'] = {
            'total_goals': len(output['events']['goals']),
            'total_fouls': len(output['events']['fouls']),
            'total_cards': len(output['events']['cards']),
            'total_substitutions': len(output['events']['substitutions']),
            'total_players': len(output['players']),
            'total_pressure_moments': len(output['events']['pressure_moments']),
            'avg_pressure_intensity': sum(pm['intensity'] for pm in output['events']['pressure_moments']) / len(output['events']['pressure_moments']) if output['events']['pressure_moments'] else 0
        }
        
        # Analysis metadata
        output['analysis_metadata'] = {
            'analyzed_at': datetime.now().isoformat(),
            'segments_analyzed': skim_data.get('focus_segments', []),
            'analysis_confidence': 'high' if len(focus_data.get('players_analyzed', [])) > 5 else 'medium',
            'video_source': os.path.basename(video_path)
        }
        
        return output

    def validate_schema(self, data: dict) -> bool:
        """Validate output matches constant schema."""
        required_keys = ['schema_version', 'match_metadata', 'teams', 'players', 
                        'events', 'statistics', 'analysis_metadata']
        
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Validate players
        for player in data['players']:
            if 'jersey_number' not in player or 'performance' not in player:
                logger.error(f"Invalid player structure: {player.get('player_id')}")
                return False
        
        # Validate events
        if not isinstance(data['events']['goals'], list):
            logger.error("Goals must be array")
            return False
        
        logger.info("✓ Schema validation passed")
        return True

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
                
                # Validation
                if not result.get('focus_segments'):
                    logger.error("Missing focus_segments")
                    await asyncio.sleep(5)
                    continue
                
                logger.info(f"✓ Pass 1 complete:")
                logger.info(f"  - Players: {len(result.get('players_detected', []))}")
                logger.info(f"  - Goals: {len(result.get('events', {}).get('goals', []))}")
                logger.info(f"  - Segments: {len(result.get('focus_segments', []))}")
                logger.info(f"  - Pressure moments: {len(result.get('events', {}).get('pressure_moments', []))}")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON error: {e}")
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
            except Exception as e:
                logger.error(f"Error: {e}")
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
        
        return {}

    async def pass2_focus_analysis(self, video_file, skim_data: dict) -> dict:
        logger.info("\n" + "=" * 60)
        logger.info("PASS 2: PERFORMANCE ANALYSIS")
        logger.info("=" * 60)
        
        if not skim_data.get('focus_segments'):
            return {}
        
        for attempt in range(3):
            try:
                logger.info(f"Attempt {attempt + 1}/3...")
                
                response = self.client.models.generate_content(
                    model=self.focus_model_name,
                    contents=[video_file, self.create_focus_prompt(skim_data)],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        top_p=0.9,
                        response_mime_type="application/json",
                        max_output_tokens=16384
                    )
                )
                
                if not response or not response.text:
                    await asyncio.sleep(5)
                    continue

                cleaned = self.fix_json(response.text)
                result = json.loads(cleaned)
                
                logger.info(f"✓ Pass 2 complete:")
                logger.info(f"  - Players analyzed: {len(result.get('players_analyzed', []))}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error: {e}")
                if attempt < 2:
                    await asyncio.sleep(15 * (attempt + 1))
        
        return {}

    def save_normalized_output(self, normalized: dict, base_name: str):
        """Save normalized output in multiple formats."""
        os.makedirs("outputfoot", exist_ok=True)
        
        # Main JSON
        json_path = f"outputfoot/{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {json_path}")
        
        # Players CSV
        players_df = pd.DataFrame([
            {
                'jersey_number': p['jersey_number'],
                'name': p['name'],
                'team': p['team'],
                'position': p['position'],
                'minutes': p['minutes_played'],
                'confidence': p['validation']['confidence'],
                'goals': p['performance']['goals'],
                'assists': p['performance']['assists'],
                'shots': p['performance']['shots'],
                'passes': p['performance']['passes'],
                'pass_accuracy': p['performance']['pass_accuracy'],
                'tackles': p['performance']['tackles'],
                'rating': p['performance']['rating'],
                'pressure_situations': p['pressure_performance']['situations_faced'],
                'composure': p['pressure_performance']['composure_rating']
            }
            for p in normalized['players']
        ])
        csv_path = f"outputfoot/{base_name}_players.csv"
        players_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved: {csv_path}")
        
        # Events CSV
        events_data = []
        for goal in normalized['events']['goals']:
            events_data.append({
                'timestamp': goal['timestamp'],
                'minute': goal['minute'],
                'type': 'GOAL',
                'player': goal['player_name'],
                'team': goal['team'],
                'details': goal['type']
            })
        for foul in normalized['events']['fouls']:
            events_data.append({
                'timestamp': foul['timestamp'],
                'minute': foul['minute'],
                'type': 'FOUL' if not foul['card'] else f"FOUL ({foul['card'].upper()})",
                'player': foul['player_name'],
                'team': foul['team'],
                'details': foul['type']
            })
        
        for sub in normalized['events']['substitutions']:
            events_data.append({
                'timestamp': sub['timestamp'],
                'minute': sub['minute'],
                'type': 'SUB',
                'player': f"{sub['player_off_name']} → {sub['player_on_name']}",
                'team': sub['team'],
                'details': 'Substitution'
            })
        
        for pressure in normalized['events']['pressure_moments']:
            events_data.append({
                'timestamp': pressure['timestamp'],
                'minute': pressure['minute'],
                'type': 'PRESSURE',
                'player': f"Intensity: {pressure['intensity']}/10",
                'team': pressure['team_under_pressure'],
                'details': f"{pressure['type']} - {pressure['outcome']}"
            })
        
        if events_data:
            events_df = pd.DataFrame(events_data)
            events_path = f"outputfoot/{base_name}_events.csv"
            events_df.to_csv(events_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved: {events_path}")

    async def run(self, video_path: str):
        """Main execution."""
        logger.info("=" * 60)
        logger.info("FOOTBALL ANALYZER - CONSTANT SCHEMA v1.0.0")
        logger.info("=" * 60)
        logger.info(f"Video: {video_path}\n")
        
        video_file = None
        try:
            # Upload
            logger.info("📤 Uploading...")
            video_file = self.client.files.upload(file=video_path)
            
            wait = 0
            while video_file.state.name == "PROCESSING" and wait < 300:
                await asyncio.sleep(5)
                wait += 5
                video_file = self.client.files.get(name=video_file.name)
                if wait % 30 == 0:
                    logger.info(f"   Processing... ({wait}s)")
            
            if video_file.state.name != "ACTIVE":
                logger.error(f"✗ Upload failed")
                return
            
            logger.info("✓ Upload complete\n")
            
            # PASS 1
            skim_data = await self.pass1_skim_analysis(video_file)
            if not skim_data:
                logger.error("✗ Pass 1 failed")
                return
            
            # PASS 2
            focus_data = await self.pass2_focus_analysis(video_file, skim_data)
            if not focus_data:
                logger.warning("⚠️ Pass 2 failed - using Pass 1 only")
                focus_data = {'players_analyzed': [], 'team_statistics': {}}
            
            # NORMALIZE TO CONSTANT SCHEMA
            logger.info("\n📋 Normalizing to constant schema...")
            normalized = self.normalize_to_schema(skim_data, focus_data, video_path)
            
            # VALIDATE SCHEMA
            if not self.validate_schema(normalized):
                logger.error("✗ Schema validation failed")
                return
            
            # SAVE
            match_id = normalized['match_metadata']['match_id']
            self.save_normalized_output(normalized, match_id)
            
            # SUMMARY
            logger.info("\n" + "=" * 60)
            logger.info("✅ ANALYSIS COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Match ID: {match_id}")
            logger.info(f"Schema: v{normalized['schema_version']}")
            logger.info(f"Score: {normalized['teams']['home']['score']}-{normalized['teams']['away']['score']}")
            logger.info(f"Players: {len(normalized['players'])}")
            logger.info(f"Events: {normalized['statistics']['match_totals']}")
            logger.info(f"Pressure Moments: {len(normalized['events']['pressure_moments'])} "
                       f"(Avg Intensity: {normalized['statistics']['match_totals']['avg_pressure_intensity']:.1f}/10)")
            logger.info(f"\n📁 Output: outputfoot/{match_id}.*")
            
        except Exception as e:
            logger.critical(f"💥 Error: {e}", exc_info=True)
        finally:
            if video_file:
                try:
                    self.client.files.delete(name=video_file.name)
                    logger.info("✓ Cleanup complete")
                except Exception as e:
                    logger.warning(f"Cleanup: {e}")


async def main():
    import sys
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("✗ Set GEMINI_API_KEY environment variable")
        return
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path>")
        return
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"✗ File not found: {video_path}")
        return
    
    analyzer = TwoPassFootballAnalyzer(api_key)
    await analyzer.run(video_path)


if __name__ == "__main__":
    asyncio.run(main())


# SCHEMA DOCUMENTATION
"""
CONSTANT OUTPUT SCHEMA v1.0.0
================================

Every video analysis produces IDENTICAL structure:

1. match_metadata
   - match_id: unique identifier
   - competition, venue, date
   - duration_seconds

2. teams
   - home: {name, score, jersey}
   - away: {name, score, jersey}

3. players[] - Array of player objects
   Each player ALWAYS has:
   - player_id, jersey_number, name, team, position
   - starter, minutes_played
   - validation: {confidence, verification_source, jersey_confirmed, name_confirmed}
   - performance: {goals, assists, shots, passes, tackles, rating, ...} [16 metrics]
   - pressure_performance: {situations_faced, composure_rating, successful_actions, errors_under_pressure, clutch_moments}

4. events
   - goals[]: timestamp, minute, player_id, team, type, score_after
   - fouls[]: timestamp, minute, player_id, team, type, card
   - cards[]: extracted from fouls with cards
   - substitutions[]: timestamp, minute, team, player_off_id, player_on_id
   - pressure_moments[]: timestamp, minute, type, team_under_pressure, intensity (1-10), outcome, key_players[]

5. statistics
   - team_home: {possession_pct, pass_accuracy_pct, shots, shots_on_target}
   - team_away: {possession_pct, pass_accuracy_pct, shots, shots_on_target}
   - match_totals: {total_goals, total_fouls, total_cards, total_substitutions, total_players, total_pressure_moments, avg_pressure_intensity}

6. analysis_metadata
   - analyzed_at: ISO timestamp
   - segments_analyzed: focus segments from Pass 1
   - analysis_confidence: high/medium/low
   - video_source: original filename

BENEFITS:
✓ Identical structure across all matches
✓ Easy to compare multiple matches
✓ Database-friendly (can insert directly)
✓ Compatible with pandas DataFrame
✓ Versioned schema (future updates backward compatible)

USAGE EXAMPLES:

# Load and compare multiple matches
import json
match1 = json.load(open('match1.json'))
match2 = json.load(open('match2.json'))

# Get top scorer across matches
all_players = match1['players'] + match2['players']
top_scorer = max(all_players, key=lambda p: p['performance']['goals'])

# Aggregate statistics
total_goals = sum(m['statistics']['match_totals']['total_goals'] 
                  for m in [match1, match2])

# Filter by confidence
verified_players = [p for p in match1['players'] 
                   if p['validation']['confidence'] == 'high']

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame([
    {
        'match_id': match1['match_metadata']['match_id'],
        'player': p['name'],
        'goals': p['performance']['goals'],
        'rating': p['performance']['rating']
    }
    for p in match1['players']
])

# Query events
goals_timeline = [
    f"{g['minute']} - {g['player_name']} ({g['team']})"
    for g in match1['events']['goals']
]

# Analyze pressure moments
high_pressure = [pm for pm in match1['events']['pressure_moments'] 
                 if pm['intensity'] >= 8]

# Get clutch performers
clutch_players = [p for p in match1['players'] 
                  if p['pressure_performance']['composure_rating'] >= 8.0
                  and p['pressure_performance']['situations_faced'] > 0]

# Pressure intensity over time
import matplotlib.pyplot as plt
pressure_df = pd.DataFrame(match1['events']['pressure_moments'])
plt.plot(pressure_df['minute'], pressure_df['intensity'])
plt.title('Match Pressure Intensity Timeline')
"""