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
        "tournament": "",
        "round": "",
        "surface": "",
        "venue": "",
        "date": "",
        "duration_seconds": 0,
        "match_type": ""
    },
    "players": {
        "player_1": {
            "name": "",
            "country": "",
            "ranking": 0,
            "seed": 0,
            "serves": "",
            "outfit": {"colors": [], "style": ""}
        },
        "player_2": {
            "name": "",
            "country": "",
            "ranking": 0,
            "seed": 0,
            "serves": "",
            "outfit": {"colors": [], "style": ""}
        }
    },
    "match_result": {
        "winner": "",
        "score": [],
        "sets_won": {"player_1": 0, "player_2": 0},
        "games_won": {"player_1": 0, "player_2": 0},
        "total_points": {"player_1": 0, "player_2": 0}
    },
    "sets": [],
    "points": [],
    "events": {
        "aces": [],
        "double_faults": [],
        "breaks_of_serve": [],
        "momentum_shifts": [],
        "pressure_points": [],
        "medical_timeouts": [],
        "challenges": []
    },
    "statistics": {
        "player_1": {},
        "player_2": {},
        "match_totals": {}
    },
    "pressure_analysis": {
        "player_1": {},
        "player_2": {},
        "key_moments": []
    },
    "analysis_metadata": {
        "analyzed_at": "",
        "segments_analyzed": [],
        "analysis_confidence": ""
    }
}

SET_SCHEMA = {
    "set_number": 0,
    "score": {"player_1": 0, "player_2": 0},
    "games": [],
    "tiebreak": False,
    "duration_minutes": 0,
    "winner": ""
}

GAME_SCHEMA = {
    "game_number": 0,
    "set_number": 0,
    "server": "",
    "score": {"player_1": 0, "player_2": 0},
    "points": [],
    "break_point": False,
    "game_point": False,
    "break_converted": False,
    "deuce_count": 0,
    "winner": ""
}

POINT_SCHEMA = {
    "point_id": "",
    "set_number": 0,
    "game_number": 0,
    "point_in_game": 0,
    "server": "",
    "score_before": {"player_1": "", "player_2": ""},
    "score_after": {"player_1": "", "player_2": ""},
    "winner": "",
    "point_type": "",
    "rally_length": 0,
    "ending_shot": "",
    "is_break_point": False,
    "is_set_point": False,
    "is_match_point": False,
    "pressure_rating": 0
}

PLAYER_STATISTICS_SCHEMA = {
    "serve": {
        "first_serve_pct": 0.0,
        "first_serve_points_won_pct": 0.0,
        "second_serve_points_won_pct": 0.0,
        "aces": 0,
        "double_faults": 0,
        "service_games_played": 0,
        "service_games_won": 0,
        "break_points_faced": 0,
        "break_points_saved": 0,
        "service_points_won": 0,
        "service_points_total": 0
    },
    "return": {
        "first_serve_return_won_pct": 0.0,
        "second_serve_return_won_pct": 0.0,
        "break_points_converted": 0,
        "break_points_opportunities": 0,
        "return_games_played": 0,
        "return_games_won": 0,
        "return_points_won": 0,
        "return_points_total": 0
    },
    "points": {
        "total_points_won": 0,
        "winners": 0,
        "unforced_errors": 0,
        "forced_errors": 0,
        "net_points": 0,
        "net_points_won": 0,
        "baseline_points_won": 0
    },
    "performance": {
        "avg_rally_length": 0.0,
        "longest_rally": 0,
        "fastest_serve_mph": 0.0,
        "avg_first_serve_speed_mph": 0.0,
        "avg_second_serve_speed_mph": 0.0,
        "dominance_ratio": 0.0
    }
}

PRESSURE_PERFORMANCE_SCHEMA = {
    "break_points": {
        "faced": 0,
        "saved": 0,
        "save_pct": 0.0,
        "conversion_pct": 0.0,
        "composure_rating": 0.0
    },
    "crucial_points": {
        "total": 0,
        "won": 0,
        "win_pct": 0.0,
        "clutch_rating": 0.0
    },
    "set_points": {
        "faced": 0,
        "saved": 0,
        "opportunities": 0,
        "converted": 0
    },
    "match_points": {
        "faced": 0,
        "saved": 0,
        "opportunities": 0,
        "converted": 0
    },
    "tiebreaks": {
        "played": 0,
        "won": 0,
        "points_won": 0,
        "points_total": 0,
        "tiebreak_rating": 0.0
    },
    "momentum": {
        "positive_swings": 0,
        "negative_swings": 0,
        "recovery_rating": 0.0,
        "mental_strength": 0.0
    }
}

EVENT_SCHEMA = {
    "ace": {
        "event_id": "",
        "timestamp": "",
        "set_number": 0,
        "game_number": 0,
        "player": "",
        "serve_speed_mph": 0.0,
        "serve_type": "",
        "score_impact": ""
    },
    "double_fault": {
        "event_id": "",
        "timestamp": "",
        "set_number": 0,
        "game_number": 0,
        "player": "",
        "pressure_situation": False,
        "score_impact": ""
    },
    "break_of_serve": {
        "event_id": "",
        "timestamp": "",
        "set_number": 0,
        "game_number": 0,
        "player_broken": "",
        "break_winner": "",
        "break_point_number": 0,
        "game_score_before": {"player_1": 0, "player_2": 0},
        "momentum_shift": ""
    },
    "pressure_point": {
        "event_id": "",
        "timestamp": "",
        "set_number": 0,
        "game_number": 0,
        "point_type": "",
        "server": "",
        "pressure_level": 0,
        "score_situation": "",
        "outcome": "",
        "winner": "",
        "rally_length": 0
    }
}

class TwoPassTennisAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.skim_model_name = 'models/gemini-2.5-flash'
        self.focus_model_name = 'models/gemini-2.5-flash'
    
    def create_skim_prompt(self) -> str:
        return """PASS 1: MATCH STRUCTURE & GAME FLOW

EXTRACT INTO THIS EXACT STRUCTURE:

{
  "match_metadata": {
    "tournament": "string",
    "round": "string (R1/R2/R3/QF/SF/F)",
    "surface": "hard/clay/grass/carpet",
    "venue": "string or null",
    "date": "YYYY-MM-DD or null",
    "duration_seconds": number,
    "match_type": "mens_singles/womens_singles/mens_doubles/womens_doubles/mixed_doubles"
  },
  "players": {
    "player_1": {
      "name": "string",
      "country": "string or null",
      "ranking": number or null,
      "seed": number or null,
      "serves": "right/left",
      "outfit": {"colors": ["string"], "style": "string"}
    },
    "player_2": {
      "name": "string",
      "country": "string or null",
      "ranking": number or null,
      "seed": number or null,
      "serves": "right/left",
      "outfit": {"colors": ["string"], "style": "string"}
    }
  },
  "match_result": {
    "winner": "player_1/player_2",
    "score": ["6-4", "3-6", "7-5"],
    "sets_won": {"player_1": number, "player_2": number}
  },
  "sets_overview": [
    {
      "set_number": number,
      "score": {"player_1": number, "player_2": number},
      "tiebreak": boolean,
      "tiebreak_score": {"player_1": number, "player_2": number} or null,
      "winner": "player_1/player_2",
      "breaks_of_serve": number
    }
  ],
  "key_events": {
    "aces": [
      {
        "timestamp": "MM:SS",
        "set": number,
        "game": number,
        "player": "player_1/player_2",
        "serve_speed_mph": number or null,
        "serve_type": "first/second"
      }
    ],
    "double_faults": [
      {
        "timestamp": "MM:SS",
        "set": number,
        "game": number,
        "player": "player_1/player_2",
        "pressure_situation": boolean
      }
    ],
    "breaks_of_serve": [
      {
        "timestamp": "MM:SS",
        "set": number,
        "game": number,
        "player_broken": "player_1/player_2",
        "break_winner": "player_1/player_2",
        "break_point_number": number,
        "score_before": {"player_1": number, "player_2": number}
      }
    ],
    "pressure_points": [
      {
        "timestamp": "MM:SS",
        "set": number,
        "game": number,
        "point_type": "break_point/set_point/match_point/deuce/game_point",
        "server": "player_1/player_2",
        "pressure_level": number (1-10),
        "score_situation": "string (e.g., '30-40', 'Ad-40')",
        "outcome": "server_won/returner_won",
        "winner": "player_1/player_2",
        "rally_length": number or null
      }
    ]
  },
  "focus_segments": [
    {
      "start": "MM:SS",
      "end": "MM:SS",
      "priority": "critical/high/medium",
      "reason": "string",
      "set": number,
      "games": [number]
    }
  ]
}

PRESSURE POINT DETECTION (CRITICAL):
✓ Break points (any score with break opportunity: 0-40, 15-40, 30-40, Ad-Out)
✓ Set points (opportunity to win set)
✓ Match points (opportunity to win match)
✓ Game points at crucial scores (5-4, 5-5, 6-5 in set)
✓ Deuce points in important games
✓ Tiebreak points (especially 6-6, match point in tiebreak)
✓ Points after momentum shifts (break, comeback)

PRESSURE LEVEL SCALE:
1-3: Regular game points
4-6: Important points (30-30, deuce, game point in close game)
7-8: Break points, crucial game points
9-10: Set points, match points, championship points

VALIDATION RULES:
✓ Use "player_1" and "player_2" consistently (not player names)
✓ All numbers must be actual numbers, not strings
✓ Timestamps in MM:SS format
✓ Include focus_segment for EACH set and tiebreak
✓ Serve speeds in mph (convert if shown in km/h)
✓ Score situation must match tennis scoring (0, 15, 30, 40, Ad)"""

    def create_focus_prompt(self, skim_data: dict) -> str:
        segments = skim_data.get('focus_segments', [])[:8]
        players = skim_data.get('players', {})
        pressure_points = skim_data.get('key_events', {}).get('pressure_points', [])
        
        player_ref = "PLAYERS:\n"
        for pid, pdata in players.items():
            name = pdata.get('name', 'Unknown')
            serves = pdata.get('serves', 'unknown')
            player_ref += f"  {pid}: {name} ({serves}-handed)\n"
        
        seg_desc = "\nANALYZE SEGMENTS:\n"
        for i, seg in enumerate(segments, 1):
            seg_desc += f"  {i}. {seg['start']}-{seg['end']}: {seg['reason']} (Set {seg['set']})\n"
        
        pressure_desc = "\nPRESSURE POINTS DETECTED:\n"
        if pressure_points:
            for pp in pressure_points[:10]:
                pressure_desc += f"  Set {pp['set']}, Game {pp['game']}: {pp['point_type']} (Pressure: {pp['pressure_level']}/10)\n"
        else:
            pressure_desc += "  None detected\n"
        
        return f"""PASS 2: DETAILED STATISTICS & PRESSURE ANALYSIS

{player_ref}
{seg_desc}
{pressure_desc}

OUTPUT THIS EXACT STRUCTURE:

{{
  "detailed_statistics": {{
    "player_1": {{
      "serve": {{
        "first_serve_pct": float (0-100),
        "first_serve_points_won_pct": float (0-100),
        "second_serve_points_won_pct": float (0-100),
        "aces": number,
        "double_faults": number,
        "service_games_played": number,
        "service_games_won": number,
        "break_points_faced": number,
        "break_points_saved": number,
        "avg_first_serve_speed_mph": float or null,
        "avg_second_serve_speed_mph": float or null,
        "fastest_serve_mph": float or null
      }},
      "return": {{
        "first_serve_return_won_pct": float (0-100),
        "second_serve_return_won_pct": float (0-100),
        "break_points_converted": number,
        "break_points_opportunities": number,
        "return_games_played": number,
        "return_games_won": number
      }},
      "points": {{
        "total_points_won": number,
        "winners": number,
        "unforced_errors": number,
        "forced_errors": number,
        "net_points": number,
        "net_points_won": number,
        "baseline_points_won": number
      }},
      "rallies": {{
        "avg_rally_length": float,
        "longest_rally": number,
        "short_rallies_won": number (0-4 shots),
        "medium_rallies_won": number (5-9 shots),
        "long_rallies_won": number (10+ shots)
      }}
    }},
    "player_2": {{
      "serve": {{}},
      "return": {{}},
      "points": {{}},
      "rallies": {{}}
    }}
  }},
  "pressure_analysis": {{
    "player_1": {{
      "break_points": {{
        "faced": number,
        "saved": number,
        "save_pct": float (0-100),
        "composure_rating": float (1-10)
      }},
      "crucial_points": {{
        "total": number,
        "won": number,
        "win_pct": float (0-100),
        "clutch_rating": float (1-10)
      }},
      "set_points": {{
        "faced": number,
        "saved": number,
        "opportunities": number,
        "converted": number
      }},
      "match_points": {{
        "faced": number,
        "saved": number,
        "opportunities": number,
        "converted": number
      }},
      "tiebreaks": {{
        "played": number,
        "won": number,
        "points_won": number,
        "points_total": number,
        "tiebreak_rating": float (1-10)
      }},
      "momentum": {{
        "positive_swings": number,
        "negative_swings": number,
        "recovery_rating": float (1-10),
        "mental_strength": float (1-10)
      }}
    }},
    "player_2": {{
      "break_points": {{}},
      "crucial_points": {{}},
      "set_points": {{}},
      "match_points": {{}},
      "tiebreaks": {{}},
      "momentum": {{}}
    }},
    "key_moments": [
      {{
        "timestamp": "MM:SS",
        "set": number,
        "game": number,
        "description": "string",
        "pressure_level": number (1-10),
        "outcome": "string",
        "impact": "game_changing/momentum_shift/routine"
      }}
    ]
  }}
}}

PRESSURE ANALYSIS INSTRUCTIONS:
- Composure rating: How well player handles pressure (1=cracks, 10=ice-cold)
- Clutch rating: Ability to win important points (1=chokes, 10=clutch performer)
- Mental strength: Overall mental resilience throughout match
- Recovery rating: Ability to bounce back after setbacks
- Tiebreak rating: Performance in tiebreaks specifically

RULES:
- All percentages: 0.0-100.0 range
- Ratings: 1.0-10.0 scale
- Use "player_1" and "player_2" consistently
- null if data unavailable"""

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
        output = json.loads(json.dumps(MATCH_SCHEMA))
        
        # Match metadata
        match_meta = skim_data.get('match_metadata', {})
        players_data = skim_data.get('players', {})
        p1_name = players_data.get('player_1', {}).get('name', 'Player 1')
        p2_name = players_data.get('player_2', {}).get('name', 'Player 2')
        
        output['match_metadata'] = {
            'match_id': f"{p1_name.replace(' ', '_')}_vs_{p2_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'tournament': match_meta.get('tournament', 'Unknown'),
            'round': match_meta.get('round', 'Unknown'),
            'surface': match_meta.get('surface', 'hard'),
            'venue': match_meta.get('venue'),
            'date': match_meta.get('date'),
            'duration_seconds': match_meta.get('duration_seconds', 0),
            'match_type': match_meta.get('match_type', 'mens_singles')
        }
        
        # Players
        output['players']['player_1'] = players_data.get('player_1', {
            'name': 'Player 1',
            'country': None,
            'ranking': 0,
            'seed': 0,
            'serves': 'right',
            'outfit': {'colors': [], 'style': ''}
        })
        output['players']['player_2'] = players_data.get('player_2', {
            'name': 'Player 2',
            'country': None,
            'ranking': 0,
            'seed': 0,
            'serves': 'right',
            'outfit': {'colors': [], 'style': ''}
        })
        
        # Match result
        match_result = skim_data.get('match_result', {})
        output['match_result'] = {
            'winner': match_result.get('winner', 'player_1'),
            'score': match_result.get('score', []),
            'sets_won': match_result.get('sets_won', {'player_1': 0, 'player_2': 0}),
            'games_won': {'player_1': 0, 'player_2': 0},
            'total_points': {'player_1': 0, 'player_2': 0}
        }
        
        # Sets overview
        sets_overview = skim_data.get('sets_overview', [])
        for set_data in sets_overview:
            set_obj = json.loads(json.dumps(SET_SCHEMA))
            set_obj['set_number'] = set_data.get('set_number', 0)
            set_obj['score'] = set_data.get('score', {'player_1': 0, 'player_2': 0})
            set_obj['tiebreak'] = set_data.get('tiebreak', False)
            set_obj['winner'] = set_data.get('winner', 'player_1')
            output['sets'].append(set_obj)
            
            # Add to games won
            output['match_result']['games_won']['player_1'] += set_obj['score']['player_1']
            output['match_result']['games_won']['player_2'] += set_obj['score']['player_2']
        
        # Events
        events = skim_data.get('key_events', {})
        
        # Aces
        for ace in events.get('aces', []):
            ace_event = json.loads(json.dumps(EVENT_SCHEMA['ace']))
            ace_event['event_id'] = f"ace_{len(output['events']['aces']) + 1}"
            ace_event['timestamp'] = ace.get('timestamp', '')
            ace_event['set_number'] = ace.get('set', 0)
            ace_event['game_number'] = ace.get('game', 0)
            ace_event['player'] = ace.get('player', 'player_1')
            ace_event['serve_speed_mph'] = ace.get('serve_speed_mph')
            ace_event['serve_type'] = ace.get('serve_type', 'first')
            output['events']['aces'].append(ace_event)
        
        # Double faults
        for df in events.get('double_faults', []):
            df_event = json.loads(json.dumps(EVENT_SCHEMA['double_fault']))
            df_event['event_id'] = f"df_{len(output['events']['double_faults']) + 1}"
            df_event['timestamp'] = df.get('timestamp', '')
            df_event['set_number'] = df.get('set', 0)
            df_event['game_number'] = df.get('game', 0)
            df_event['player'] = df.get('player', 'player_1')
            df_event['pressure_situation'] = df.get('pressure_situation', False)
            output['events']['double_faults'].append(df_event)
        
        # Breaks of serve
        for brk in events.get('breaks_of_serve', []):
            brk_event = json.loads(json.dumps(EVENT_SCHEMA['break_of_serve']))
            brk_event['event_id'] = f"break_{len(output['events']['breaks_of_serve']) + 1}"
            brk_event['timestamp'] = brk.get('timestamp', '')
            brk_event['set_number'] = brk.get('set', 0)
            brk_event['game_number'] = brk.get('game', 0)
            brk_event['player_broken'] = brk.get('player_broken', 'player_1')
            brk_event['break_winner'] = brk.get('break_winner', 'player_2')
            brk_event['break_point_number'] = brk.get('break_point_number', 1)
            brk_event['game_score_before'] = brk.get('score_before', {'player_1': 0, 'player_2': 0})
            output['events']['breaks_of_serve'].append(brk_event)
        
        # Pressure points
        for pp in events.get('pressure_points', []):
            pp_event = json.loads(json.dumps(EVENT_SCHEMA['pressure_point']))
            pp_event['event_id'] = f"pressure_{len(output['events']['pressure_points']) + 1}"
            pp_event['timestamp'] = pp.get('timestamp', '')
            pp_event['set_number'] = pp.get('set', 0)
            pp_event['game_number'] = pp.get('game', 0)
            pp_event['point_type'] = pp.get('point_type', 'break_point')
            pp_event['server'] = pp.get('server', 'player_1')
            pp_event['pressure_level'] = pp.get('pressure_level', 5)
            pp_event['score_situation'] = pp.get('score_situation', '')
            pp_event['outcome'] = pp.get('outcome', '')
            pp_event['winner'] = pp.get('winner', 'player_1')
            pp_event['rally_length'] = pp.get('rally_length')
            output['events']['pressure_points'].append(pp_event)
        
        # Statistics
        detailed_stats = focus_data.get('detailed_statistics', {})
        
        for player_id in ['player_1', 'player_2']:
            player_stats = detailed_stats.get(player_id, {})
            
            output['statistics'][player_id] = json.loads(json.dumps(PLAYER_STATISTICS_SCHEMA))
            
            # Serve stats
            serve = player_stats.get('serve', {})
            output['statistics'][player_id]['serve'] = {
                'first_serve_pct': serve.get('first_serve_pct', 0.0),
                'first_serve_points_won_pct': serve.get('first_serve_points_won_pct', 0.0),
                'second_serve_points_won_pct': serve.get('second_serve_points_won_pct', 0.0),
                'aces': serve.get('aces', 0),
                'double_faults': serve.get('double_faults', 0),
                'service_games_played': serve.get('service_games_played', 0),
                'service_games_won': serve.get('service_games_won', 0),
                'break_points_faced': serve.get('break_points_faced', 0),
                'break_points_saved': serve.get('break_points_saved', 0),
                'service_points_won': 0,
                'service_points_total': 0
            }
            
            # Return stats
            ret = player_stats.get('return', {})
            output['statistics'][player_id]['return'] = {
                'first_serve_return_won_pct': ret.get('first_serve_return_won_pct', 0.0),
                'second_serve_return_won_pct': ret.get('second_serve_return_won_pct', 0.0),
                'break_points_converted': ret.get('break_points_converted', 0),
                'break_points_opportunities': ret.get('break_points_opportunities', 0),
                'return_games_played': ret.get('return_games_played', 0),
                'return_games_won': ret.get('return_games_won', 0),
                'return_points_won': 0,
                'return_points_total': 0
            }
            
            # Points stats
            pts = player_stats.get('points', {})
            output['statistics'][player_id]['points'] = {
                'total_points_won': pts.get('total_points_won', 0),
                'winners': pts.get('winners', 0),
                'unforced_errors': pts.get('unforced_errors', 0),
                'forced_errors': pts.get('forced_errors', 0),
                'net_points': pts.get('net_points', 0),
                'net_points_won': pts.get('net_points_won', 0),
                'baseline_points_won': pts.get('baseline_points_won', 0)
            }
            
            # Update match result total points
            output['match_result']['total_points'][player_id] = pts.get('total_points_won', 0)
            
            # Performance stats
            rallies = player_stats.get('rallies', {})
            output['statistics'][player_id]['performance'] = {
                'avg_rally_length': rallies.get('avg_rally_length', 0.0),
                'longest_rally': rallies.get('longest_rally', 0),
                'fastest_serve_mph': serve.get('fastest_serve_mph', 0.0),
                'avg_first_serve_speed_mph': serve.get('avg_first_serve_speed_mph', 0.0),
                'avg_second_serve_speed_mph': serve.get('avg_second_serve_speed_mph', 0.0),
                'dominance_ratio': 0.0
            }
        
        # Pressure analysis
        pressure_analysis = focus_data.get('pressure_analysis', {})
        
        for player_id in ['player_1', 'player_2']:
            player_pressure = pressure_analysis.get(player_id, {})
            
            output['pressure_analysis'][player_id] = json.loads(json.dumps(PRESSURE_PERFORMANCE_SCHEMA))
            
            # Break points
            bp = player_pressure.get('break_points', {})
            output['pressure_analysis'][player_id]['break_points'] = {
                'faced': bp.get('faced', 0),
                'saved': bp.get('saved', 0),
                'save_pct': bp.get('save_pct', 0.0),
                'conversion_pct': 0.0,
                'composure_rating': bp.get('composure_rating', 0.0)
            }
            
            # Crucial points
            cp = player_pressure.get('crucial_points', {})
            output['pressure_analysis'][player_id]['crucial_points'] = {
                'total': cp.get('total', 0),
                'won': cp.get('won', 0),
                'win_pct': cp.get('win_pct', 0.0),
                'clutch_rating': cp.get('clutch_rating', 0.0)
            }
            
            # Set points
            sp = player_pressure.get('set_points', {})
            output['pressure_analysis'][player_id]['set_points'] = {
                'faced': sp.get('faced', 0),
                'saved': sp.get('saved', 0),
                'opportunities': sp.get('opportunities', 0),
                'converted': sp.get('converted', 0)
            }
            
            # Match points
            mp = player_pressure.get('match_points', {})
            output['pressure_analysis'][player_id]['match_points'] = {
                'faced': mp.get('faced', 0),
                'saved': mp.get('saved', 0),
                'opportunities': mp.get('opportunities', 0),
                'converted': mp.get('converted', 0)
            }
            
            # Tiebreaks
            tb = player_pressure.get('tiebreaks', {})
            output['pressure_analysis'][player_id]['tiebreaks'] = {
                'played': tb.get('played', 0),
                'won': tb.get('won', 0),
                'points_won': tb.get('points_won', 0),
                'points_total': tb.get('points_total', 0),
                'tiebreak_rating': tb.get('tiebreak_rating', 0.0)
            }
            
            # Momentum
            mom = player_pressure.get('momentum', {})
            output['pressure_analysis'][player_id]['momentum'] = {
                'positive_swings': mom.get('positive_swings', 0),
                'negative_swings': mom.get('negative_swings', 0),
                'recovery_rating': mom.get('recovery_rating', 0.0),
                'mental_strength': mom.get('mental_strength', 0.0)
            }
        
        # Key moments
        output['pressure_analysis']['key_moments'] = pressure_analysis.get('key_moments', [])
        
        # Match totals
        p1_total_points = output['match_result']['total_points']['player_1'] or 0
        p2_total_points = output['match_result']['total_points']['player_2'] or 0
        
        output['statistics']['match_totals'] = {
            'total_games': output['match_result']['games_won']['player_1'] + output['match_result']['games_won']['player_2'],
            'total_points': p1_total_points + p2_total_points,
            'total_aces': len(output['events']['aces']),
            'total_double_faults': len(output['events']['double_faults']),
            'total_breaks': len(output['events']['breaks_of_serve']),
            'total_pressure_points': len(output['events']['pressure_points']),
            'avg_pressure_level': sum(pp['pressure_level'] for pp in output['events']['pressure_points']) / len(output['events']['pressure_points']) if output['events']['pressure_points'] else 0.0,
            'match_quality_rating': 0.0
        }
        
        # Calculate match quality rating (1-10 scale)
        quality_factors = []
        if output['statistics']['match_totals']['total_pressure_points'] > 10:
            quality_factors.append(8.0)
        if len(output['sets']) >= 3:
            quality_factors.append(7.5)
        if output['statistics']['match_totals']['avg_pressure_level'] >= 7.0:
            quality_factors.append(8.5)
        
        output['statistics']['match_totals']['match_quality_rating'] = sum(quality_factors) / len(quality_factors) if quality_factors else 6.0
        
        # Analysis metadata
        output['analysis_metadata'] = {
            'analyzed_at': datetime.now().isoformat(),
            'segments_analyzed': skim_data.get('focus_segments', []),
            'analysis_confidence': 'high' if len(output['events']['pressure_points']) > 5 else 'medium',
            'video_source': os.path.basename(video_path)
        }
        
        return output

    def validate_schema(self, data: dict) -> bool:
        """Validate output matches constant schema."""
        required_keys = ['schema_version', 'match_metadata', 'players', 'match_result',
                        'sets', 'events', 'statistics', 'pressure_analysis', 'analysis_metadata']
        
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Validate players
        if 'player_1' not in data['players'] or 'player_2' not in data['players']:
            logger.error("Missing player data")
            return False
        
        # Validate pressure analysis
        if 'player_1' not in data['pressure_analysis'] or 'player_2' not in data['pressure_analysis']:
            logger.error("Missing pressure analysis")
            return False
        
        logger.info("✓ Schema validation passed")
        return True

    async def pass1_skim_analysis(self, video_file) -> dict:
        logger.info("=" * 60)
        logger.info("PASS 1: MATCH STRUCTURE ANALYSIS")
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
                logger.info(f"  - Sets: {len(result.get('sets_overview', []))}")
                logger.info(f"  - Aces: {len(result.get('key_events', {}).get('aces', []))}")
                logger.info(f"  - Breaks: {len(result.get('key_events', {}).get('breaks_of_serve', []))}")
                logger.info(f"  - Pressure points: {len(result.get('key_events', {}).get('pressure_points', []))}")
                logger.info(f"  - Focus segments: {len(result.get('focus_segments', []))}")
                
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
        logger.info("PASS 2: STATISTICS & PRESSURE ANALYSIS")
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
                logger.info(f"  - Statistics extracted for both players")
                logger.info(f"  - Pressure analysis complete")
                
                return result
                
            except Exception as e:
                logger.error(f"Error: {e}")
                if attempt < 2:
                    await asyncio.sleep(15 * (attempt + 1))
        
        return {}

    def save_normalized_output(self, normalized: dict, base_name: str):
        """Save normalized output in multiple formats."""
        os.makedirs("outputtennis", exist_ok=True)
        
        # Main JSON
        json_path = f"outputtennis/{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {json_path}")
        
        # Player statistics CSV
        stats_data = []
        for player_id in ['player_1', 'player_2']:
            player = normalized['players'][player_id]
            stats = normalized['statistics'][player_id]
            pressure = normalized['pressure_analysis'][player_id]
            
            stats_data.append({
                'player': player['name'],
                'country': player.get('country', ''),
                'ranking': player.get('ranking', 0),
                'serves': player['serves'],
                # Serve stats
                'first_serve_pct': stats['serve']['first_serve_pct'],
                'first_serve_won_pct': stats['serve']['first_serve_points_won_pct'],
                'second_serve_won_pct': stats['serve']['second_serve_points_won_pct'],
                'aces': stats['serve']['aces'],
                'double_faults': stats['serve']['double_faults'],
                'service_games_won': stats['serve']['service_games_won'],
                'break_points_saved_pct': stats['serve']['break_points_saved'] / stats['serve']['break_points_faced'] * 100 if stats['serve']['break_points_faced'] > 0 else 0,
                # Return stats
                'return_points_won': stats['return']['return_points_won'],
                'break_points_converted': stats['return']['break_points_converted'],
                'return_games_won': stats['return']['return_games_won'],
                # Points
                'total_points_won': stats['points']['total_points_won'],
                'winners': stats['points']['winners'],
                'unforced_errors': stats['points']['unforced_errors'],
                # Pressure
                'break_point_composure': pressure['break_points']['composure_rating'],
                'clutch_rating': pressure['crucial_points']['clutch_rating'],
                'mental_strength': pressure['momentum']['mental_strength'],
                'tiebreak_rating': pressure['tiebreaks']['tiebreak_rating']
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_csv = f"outputtennis/{base_name}_statistics.csv"
        stats_df.to_csv(stats_csv, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved: {stats_csv}")
        
        # Events timeline CSV
        events_data = []
        
        for ace in normalized['events']['aces']:
            events_data.append({
                'timestamp': ace['timestamp'],
                'set': ace['set_number'],
                'game': ace['game_number'],
                'type': 'ACE',
                'player': ace['player'],
                'details': f"{ace.get('serve_speed_mph', 'N/A')} mph" if ace.get('serve_speed_mph') else 'N/A'
            })
        
        for df in normalized['events']['double_faults']:
            events_data.append({
                'timestamp': df['timestamp'],
                'set': df['set_number'],
                'game': df['game_number'],
                'type': 'DOUBLE FAULT',
                'player': df['player'],
                'details': 'Under pressure' if df['pressure_situation'] else 'Regular'
            })
        
        for brk in normalized['events']['breaks_of_serve']:
            events_data.append({
                'timestamp': brk['timestamp'],
                'set': brk['set_number'],
                'game': brk['game_number'],
                'type': 'BREAK OF SERVE',
                'player': brk['break_winner'],
                'details': f"Broke {brk['player_broken']}"
            })
        
        for pp in normalized['events']['pressure_points']:
            events_data.append({
                'timestamp': pp['timestamp'],
                'set': pp['set_number'],
                'game': pp['game_number'],
                'type': f"PRESSURE ({pp['point_type']})",
                'player': pp['winner'],
                'details': f"Level {pp['pressure_level']}/10 - {pp['score_situation']}"
            })
        
        if events_data:
            events_df = pd.DataFrame(events_data)
            events_df = events_df.sort_values(['set', 'game'])
            events_csv = f"outputtennis/{base_name}_events.csv"
            events_df.to_csv(events_csv, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved: {events_csv}")
        
        # Pressure analysis CSV
        pressure_data = []
        for player_id in ['player_1', 'player_2']:
            player = normalized['players'][player_id]
            pressure = normalized['pressure_analysis'][player_id]
            
            pressure_data.append({
                'player': player['name'],
                'bp_faced': pressure['break_points']['faced'],
                'bp_saved': pressure['break_points']['saved'],
                'bp_save_pct': pressure['break_points']['save_pct'],
                'bp_composure': pressure['break_points']['composure_rating'],
                'crucial_points_won': pressure['crucial_points']['won'],
                'crucial_points_total': pressure['crucial_points']['total'],
                'clutch_rating': pressure['crucial_points']['clutch_rating'],
                'set_points_faced': pressure['set_points']['faced'],
                'set_points_saved': pressure['set_points']['saved'],
                'match_points_faced': pressure['match_points']['faced'],
                'match_points_saved': pressure['match_points']['saved'],
                'tiebreaks_won': pressure['tiebreaks']['won'],
                'tiebreaks_played': pressure['tiebreaks']['played'],
                'tiebreak_rating': pressure['tiebreaks']['tiebreak_rating'],
                'mental_strength': pressure['momentum']['mental_strength'],
                'recovery_rating': pressure['momentum']['recovery_rating']
            })
        
        pressure_df = pd.DataFrame(pressure_data)
        pressure_csv = f"outputtennis/{base_name}_pressure.csv"
        pressure_df.to_csv(pressure_csv, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved: {pressure_csv}")

    async def run(self, video_path: str):
        """Main execution."""
        logger.info("=" * 60)
        logger.info("TENNIS ANALYZER - CONSTANT SCHEMA v1.0.0")
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
                focus_data = {'detailed_statistics': {}, 'pressure_analysis': {}}
            
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
            logger.info(f"Winner: {normalized['players'][normalized['match_result']['winner']]['name']}")
            logger.info(f"Score: {' '.join(normalized['match_result']['score'])}")
            logger.info(f"Sets: {normalized['match_result']['sets_won']}")
            logger.info(f"Total games: {normalized['statistics']['match_totals']['total_games']}")
            logger.info(f"Aces: {normalized['statistics']['match_totals']['total_aces']}")
            logger.info(f"Breaks: {normalized['statistics']['match_totals']['total_breaks']}")
            logger.info(f"Pressure points: {normalized['statistics']['match_totals']['total_pressure_points']} "
                       f"(Avg Level: {normalized['statistics']['match_totals']['avg_pressure_level']:.1f}/10)")
            logger.info(f"Match quality: {normalized['statistics']['match_totals']['match_quality_rating']:.1f}/10")
            logger.info(f"\n📁 Output: outputtennis/{match_id}.*")
            
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
        print("Usage: python tennis_analyzer.py <video_path>")
        return
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"✗ File not found: {video_path}")
        return
    
    analyzer = TwoPassTennisAnalyzer(api_key)
    await analyzer.run(video_path)


if __name__ == "__main__":
    asyncio.run(main())


# SCHEMA DOCUMENTATION
"""
CONSTANT OUTPUT SCHEMA v1.0.0 - TENNIS
========================================

Every tennis match analysis produces IDENTICAL structure:

1. match_metadata
   - match_id: unique identifier
   - tournament, round, surface, venue, date
   - duration_seconds, match_type

2. players
   - player_1: {name, country, ranking, seed, serves, outfit}
   - player_2: {name, country, ranking, seed, serves, outfit}

3. match_result
   - winner: player_1 or player_2
   - score: ["6-4", "3-6", "7-5"]
   - sets_won: {player_1: 2, player_2: 1}
   - games_won: {player_1: 16, player_2: 15}
   - total_points: {player_1: 145, player_2: 138}

4. sets[] - Array of set objects
   Each set has:
   - set_number, score, tiebreak, winner
   - games[] array with game-by-game breakdown

5. events
   - aces[]: timestamp, set, game, player, serve_speed_mph
   - double_faults[]: timestamp, set, game, player, pressure_situation
   - breaks_of_serve[]: timestamp, set, game, player_broken, break_winner
   - pressure_points[]: timestamp, set, game, point_type, pressure_level (1-10), outcome
   - momentum_shifts[]: when momentum changed significantly
   - medical_timeouts[], challenges[]

6. statistics (per player)
   - serve: first_serve_pct, aces, double_faults, break_points_saved, etc.
   - return: return_won_pct, break_points_converted, etc.
   - points: total_points_won, winners, unforced_errors, etc.
   - performance: avg_rally_length, fastest_serve, dominance_ratio

7. pressure_analysis (per player) ⭐ KEY FEATURE
   - break_points: {faced, saved, save_pct, composure_rating (1-10)}
   - crucial_points: {total, won, win_pct, clutch_rating (1-10)}
   - set_points: {faced, saved, opportunities, converted}
   - match_points: {faced, saved, opportunities, converted}
   - tiebreaks: {played, won, points_won, tiebreak_rating (1-10)}
   - momentum: {positive_swings, negative_swings, recovery_rating, mental_strength (1-10)}
   - key_moments[]: Critical points with pressure_level and impact

8. analysis_metadata
   - analyzed_at: ISO timestamp
   - segments_analyzed: focus segments from Pass 1
   - analysis_confidence: high/medium/low
   - video_source: original filename

TENNIS-SPECIFIC PRESSURE METRICS:
==================================

1. Break Point Composure (1-10): How well player handles break points
   - 9-10: Ice cold, almost always saves/converts
   - 7-8: Very good under pressure
   - 5-6: Average performance
   - 3-4: Struggles under pressure
   - 1-2: Crumbles in crucial moments

2. Clutch Rating (1-10): Performance in important points
   - Measures: Set points, game points at crucial scores, momentum points
   - Higher = better at winning the points that matter most

3. Mental Strength (1-10): Overall psychological resilience
   - Recovery from bad games
   - Performance when behind
   - Ability to close out matches

4. Tiebreak Rating (1-10): Specific tiebreak performance
   - Tiebreaks are high-pressure situations
   - Separate metric due to unique scoring

5. Pressure Level (1-10): For individual points
   - 1-3: Regular points
   - 4-6: Important (30-30, deuce, game point)
   - 7-8: Critical (break point, crucial game)
   - 9-10: Match-defining (set/match point, championship point)

USAGE EXAMPLES:
===============

# Load match data
import json
match = json.load(open('federer_vs_nadal_20250102_143022.json'))

# Find clutch performer
p1_clutch = match['pressure_analysis']['player_1']['crucial_points']['clutch_rating']
p2_clutch = match['pressure_analysis']['player_2']['crucial_points']['clutch_rating']
clutch_player = match['players']['player_1']['name'] if p1_clutch > p2_clutch else match['players']['player_2']['name']

# Analyze break points
for player_id in ['player_1', 'player_2']:
    bp = match['pressure_analysis'][player_id]['break_points']
    print(f"{match['players'][player_id]['name']}: {bp['saved']}/{bp['faced']} BP saved ({bp['save_pct']:.1f}%)")

# Get high-pressure moments
high_pressure = [pp for pp in match['events']['pressure_points'] if pp['pressure_level'] >= 8]

# Compare mental strength
import pandas as pd
mental_comparison = pd.DataFrame([
    {
        'player': match['players'][pid]['name'],
        'mental_strength': match['pressure_analysis'][pid]['momentum']['mental_strength'],
        'composure': match['pressure_analysis'][pid]['break_points']['composure_rating'],
        'clutch': match['pressure_analysis'][pid]['crucial_points']['clutch_rating']
    }
    for pid in ['player_1', 'player_2']
])

# Timeline of momentum
momentum_events = match['events']['momentum_shifts']

# Match quality assessment
quality = match['statistics']['match_totals']['match_quality_rating']

# Pressure points timeline
pressure_timeline = pd.DataFrame(match['events']['pressure_points'])
pressure_timeline['minute'] = pressure_timeline['timestamp'].apply(lambda x: int(x.split(':')[0]))

# Service statistics comparison
for pid in ['player_1', 'player_2']:
    serve = match['statistics'][pid]['serve']
    print(f"{match['players'][pid]['name']}: {serve['aces']} aces, {serve['first_serve_pct']:.1f}% first serves")

# Identify match-turning points
turning_points = [km for km in match['pressure_analysis']['key_moments'] 
                  if km['impact'] == 'game_changing']
"""