# THIS IS THE CORRECT SCRIPT TO USE
from google import genai
import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple
import logging
import time
import asyncio
from pathlib import Path
import google.api_core.exceptions  # <-- Required import for retry logic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoPassBasketballAnalyzer:
    def __init__(self, api_key: str):
        """Initialize analyzer with Gemini API."""
        self.client = genai.Client(api_key=api_key)
        self.skim_model_name = 'models/gemini-2.5-flash'
        self.focus_model_name = 'models/gemini-2.5-flash'
    
    # --- Prompts Define the JSON Structure ---
    def create_skim_prompt(self) -> str:
        """PASS 1: Lightweight context-gathering prompt."""
        # This prompt clearly defines the desired JSON structure for Pass 1
        return f"""PASS 1: LIGHTWEIGHT VIDEO MAPPING
Analyze this basketball game video at LOW DETAIL to create a structural map.

YOUR TASK (Fast Scan Only):
1. Identify team names from jerseys/scoreboard
2. Identify which quarters are visible in the video
3. Find ALL player substitutions (when players enter/exit)
4. Detect high-pressure moments (close score + late quarter)
5. Identify ALL unique players (jersey numbers/names)
6. Mark EVERY key event timestamp (shots, fouls, blocks, etc.)

IMPORTANT: Be COMPREHENSIVE in detecting key moments. Track EVERY significant play.

OUTPUT FORMAT (JSON only):
{{
  "video_structure": {{
    "team_a": "Auto-detected Team A",
    "team_b": "Auto-detected Team B",
    "match_type": "Regular Season/Playoffs/etc (if detectable)",
    "match_year": "2024",
    "total_duration_seconds": 600,
    "quarters_present": ["Q4"],
    "game_time_range": "Q4 10:00-0:00"
  }},
  "players_detected": [
    {{
      "player_id": "#23 LeBron",
      "team": "Lakers",
      "visible_quarters": ["Q4"],
      "is_starter": true,
      "estimated_minutes": 10
    }}
  ],
  "key_moments": [
    {{
      "timestamp": "2:45-0:00",
      "quarter": "Q4",
      "moment_type": "high_pressure",
      "reason": "Close score, final 3 minutes",
      "score_differential": "Within 3 points"
    }},
    {{
      "timestamp": "8:30",
      "quarter": "Q4", 
      "moment_type": "substitution",
      "players_in": ["#6"],
      "players_out": ["#12"]
    }},
    {{
      "timestamp": "0:17",
      "quarter": "Q4",
      "moment_type": "foul",
      "players_involved": ["#4 Shumpert"],
      "event_details": "Foul by Cavaliers #4 on Warriors #34"
    }}
  ],
  "focus_segments": [
    {{
      "start_time": "2:45",
      "end_time": "0:00",
      "quarter": "Q4",
      "priority": "HIGH",
      "reason": "Crunch time with close score - needs detailed analysis"
    }}
  ]
}}

CRITICAL: Track EVERY significant play (shots, fouls, blocks, turnovers, etc.) with timestamps.
Return valid JSON only."""

    def create_focus_prompt(self, skim_data: Dict) -> str:
        """PASS 2: Deep analysis prompt targeting specific segments."""
        # This prompt clearly defines the desired JSON structure for Pass 2
        focus_segments = skim_data.get('focus_segments', [])
        players = skim_data.get('players_detected', [])
        video_structure = skim_data.get('video_structure', {})
        
        segments_desc = "\n".join([
            f"- {seg['start_time']} to {seg['end_time']} ({seg['quarter']}): {seg['reason']}"
            for seg in focus_segments
        ])
        
        player_list = ", ".join([p['player_id'] for p in players])
        team_a = video_structure.get('team_a', 'Team A')
        team_b = video_structure.get('team_b', 'Team B')
        
        return f"""PASS 2: FOCUSED DEEP ANALYSIS
Based on Pass 1 mapping, analyze ONLY these specific segments in MAXIMUM DETAIL:

Teams: {team_a} vs {team_b}

FOCUS SEGMENTS:
{segments_desc}

KNOWN PLAYERS: {player_list}

YOUR TASK (Detailed Analysis - BE EXTREMELY THOROUGH):
For each player in the FOCUS SEGMENTS:

1. GRANULAR STATISTICS (track EVERY play):
   - Track EVERY shot attempt with type and result
   - Count EVERY assist, turnover, rebound
   - Record EVERY steal, block, foul
   - Rate shot quality (1-10) and decision making (1-10)

2. PRESSURE PERFORMANCE (for high-pressure segments):
   - Score ALL statistics under pressure
   - Mental toughness rating (1-10) with detailed explanation
   - Clutch factor with SPECIFIC play descriptions
   - List EVERY critical play made/missed

3. TEMPORAL ANALYSIS (detailed behavioral tracking):
   - Player rhythm and confidence evolution
   - Specific fatigue indicators
   - Performance consistency with examples
   - Emotional state and body language

OUTPUT FORMAT (JSON - BE COMPREHENSIVE):
{{
  "analysis_metadata": {{
    "segments_analyzed": ["Q4 2:45-0:00"],
    "total_analysis_duration": "2.75 minutes",
    "pressure_segments": ["Q4 2:45-0:00"]
  }},
  "players": [
    {{
      "player_id": "#23 LeBron",
      "team": "Lakers",
      "position": "SF",
      "role": "Starter",
      "analyzed_segments": [
        {{
          "timeframe": "Q4 2:45-0:00",
          "segment_type": "high_pressure",
          "minutes_played": 2.75,
          "stats": {{
            "fgm": 2, "fga": 4, "fg_pct": 50.0,
            "3pm": 1, "3pa": 2, "3p_pct": 50.0,
            "ftm": 2, "fta": 2, "ft_pct": 100.0,
            "pts": 7, "ast": 1, "tov": 0,
            "oreb": 0, "dreb": 2, "reb": 2,
            "stl": 1, "blk": 0, "pf": 0
          }},
          "pressure_metrics": {{
            "pressure_intensity": 10,
            "mental_toughness": 9,
            "clutch_factor": "Hit game-winning three with 12 seconds left after two defensive stops",
            "critical_plays": 3,
            "shot_quality": 8,
            "decision_making": 9
          }},
          "temporal_analysis": {{
            "rhythm_confidence": "Started tentative, grew confident after first make at 2:15",
            "fatigue_indicators": "Heavy breathing at 1:00 mark, but maintained explosiveness",
            "performance_consistency": "Clutch execution in final minute after missing two shots earlier"
          }}
        }}
      ],
      "segment_totals": {{
        "fgm": 2, "fga": 4, "fg_pct": 50.0,
        "pts": 7, "ast": 1, "reb": 2
      }}
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
- Analyze EVERY SINGLE PLAY in the focus segments
- Provide SPECIFIC descriptions, not generic statements
- Include temporal_analysis with detailed observations
- Track actual plays with timestamps when possible
- Be as comprehensive as the Pass 1 key_moments detection
- Return valid JSON only"""
    # --- End of Prompts ---

    async def pass1_skim_analysis(self, video_file: genai.types.File) -> Dict:
        """PASS 1: Fast structural analysis with retry logic."""
        logger.info("=" * 60)
        logger.info("PASS 1: SKIM ANALYSIS (Context Gathering)")
        logger.info("=" * 60)
        
        prompt = self.create_skim_prompt()
        logger.info("Running lightweight structural analysis...")
        logger.info("→ Detecting teams, quarters, players, and key moments")
        
        max_retries = 3
        retry_delay_seconds = 5
        last_exception = None
        response = None # Define response outside the loop

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.skim_model_name,
                    contents=[video_file, prompt],
                    # You can add generation_config here if needed
                    # generation_config=genai.types.GenerationConfig(
                    #     temperature=0.3,
                    #     top_p=0.95
                    # )
                )
                
                # Success - parse and return
                # Make sure response has text before processing
                if not response or not hasattr(response, 'text') or not response.text:
                     logger.error("Pass 1: Received empty response from API.")
                     return {}

                cleaned = response.text.strip().replace('```json', '').replace('```', '').strip()
                result = json.loads(cleaned)
                
                players_found = len(result.get('players_detected', []))
                focus_segments = len(result.get('focus_segments', []))
                key_moments = len(result.get('key_moments', []))
                video_structure = result.get('video_structure', {})
                
                logger.info(f"✓ Skim complete:")
                logger.info(f"  - Teams: {video_structure.get('team_a', 'N/A')} vs {video_structure.get('team_b', 'N/A')}")
                logger.info(f"  - {players_found} players detected")
                logger.info(f"  - {key_moments} key moments tracked")
                logger.info(f"  - {focus_segments} segments flagged for deep analysis")
                logger.info(f"  - Quarters: {', '.join(video_structure.get('quarters_present', []))}")
                
                return result
                
            except (google.api_core.exceptions.ServiceUnavailable, 
                    google.api_core.exceptions.ResourceExhausted) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Pass 1: Model overloaded (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay_seconds}s...")
                    await asyncio.sleep(retry_delay_seconds)
                    retry_delay_seconds *= 2
                else:
                    logger.error(f"Pass 1 failed after {max_retries} attempts: {e}")
                    return {}
            except json.JSONDecodeError as e:
                logger.error(f"Pass 1 JSON parsing error: {e}")
                # Log the raw response text if parsing fails
                if response and hasattr(response, 'text'):
                     logger.error(f"Raw response: {response.text[:500]}...") # Log beginning of text
                else:
                     logger.error("Raw response was empty or not available.")
                return {}
            except Exception as e:
                logger.error(f"Pass 1 unexpected error: {e}", exc_info=True)
                return {}
        
        # If we exhausted retries
        logger.error(f"Pass 1 failed after all retries: {last_exception}")
        return {}

    async def pass2_focus_analysis(self, video_file: genai.types.File, skim_data: Dict) -> Dict:
        """PASS 2: Deep analysis with enhanced prompting."""
        logger.info("\n" + "=" * 60)
        logger.info("PASS 2: FOCUS ANALYSIS (Deep Dive)")
        logger.info("=" * 60)
        
        focus_segments = skim_data.get('focus_segments', [])
        if not focus_segments:
            logger.warning("No focus segments identified - skipping Pass 2")
            return {}
        
        logger.info(f"Analyzing {len(focus_segments)} high-priority segments:")
        for seg in focus_segments:
            logger.info(f"  → {seg['quarter']} {seg['start_time']}-{seg['end_time']} ({seg['priority']})")
        
        prompt = self.create_focus_prompt(skim_data)
        logger.info("Running deep analysis on focus segments...")
        
        max_retries = 3
        retry_delay_seconds = 5
        last_exception = None
        response = None # Define response outside the loop

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.focus_model_name,
                    contents=[video_file, prompt],
                     # You can add generation_config here if needed
                    # generation_config=genai.types.GenerationConfig(
                    #     temperature=0.3,
                    #     top_p=0.95
                    # )
                )
                
                # Success - parse and return
                # Make sure response has text before processing
                if not response or not hasattr(response, 'text') or not response.text:
                     logger.error("Pass 2: Received empty response from API.")
                     return {}

                cleaned = response.text.strip().replace('```json', '').replace('```', '').strip()
                result = json.loads(cleaned)
                
                players_analyzed = len(result.get('players', []))
                logger.info(f"✓ Focus analysis complete:")
                logger.info(f"  - {players_analyzed} players analyzed in detail")
                
                return result
                
            except (google.api_core.exceptions.ServiceUnavailable,
                    google.api_core.exceptions.ResourceExhausted) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Pass 2: Model overloaded (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay_seconds}s...")
                    await asyncio.sleep(retry_delay_seconds)
                    retry_delay_seconds *= 2
                else:
                    logger.error(f"Pass 2 failed after {max_retries} attempts: {e}")
                    return {}
            except json.JSONDecodeError as e:
                logger.error(f"Pass 2 JSON parsing error: {e}")
                 # Log the raw response text if parsing fails
                if response and hasattr(response, 'text'):
                     logger.error(f"Raw response: {response.text[:500]}...") # Log beginning of text
                else:
                     logger.error("Raw response was empty or not available.")
                return {}
            except Exception as e:
                logger.error(f"Pass 2 unexpected error: {e}", exc_info=True)
                return {}
        
        # If we exhausted retries
        logger.error(f"Pass 2 failed after all retries: {last_exception}")
        return {}

    def merge_analysis_results(self, skim_data: Dict, focus_data: Dict) -> Dict:
        """Combine Pass 1 and Pass 2 results."""
        logger.info("\n" + "=" * 60)
        logger.info("MERGING ANALYSIS RESULTS")
        logger.info("=" * 60)
        
        merged = {
            'game_info': {
                'video_structure': skim_data.get('video_structure', {}),
                'analysis_metadata': focus_data.get('analysis_metadata', {}),
                'key_moments': skim_data.get('key_moments', [])
            },
            'players': []
        }
        
        skim_players = {p['player_id']: p for p in skim_data.get('players_detected', [])}
        focus_players = {p['player_id']: p for p in focus_data.get('players', [])}
        
        for player_id in skim_players.keys():
            player_data = {
                'player_id': player_id,
                'team': skim_players[player_id].get('team'),
                'detection_info': skim_players[player_id],
                'detailed_analysis': focus_players.get(player_id, {})
            }
            merged['players'].append(player_data)
        
        logger.info(f"✓ Merged {len(merged['players'])} player profiles")
        logger.info(f"✓ Captured {len(merged['game_info']['key_moments'])} key moments")
        return merged

    def convert_to_csv(self, merged_data: Dict) -> pd.DataFrame:
        """Convert merged analysis to CSV format."""
        if not merged_data or 'players' not in merged_data:
            return pd.DataFrame()
        
        logger.info("Converting results to CSV...")
        rows = []
        game_info = merged_data.get('game_info', {})
        video_structure = game_info.get('video_structure', {})
        analysis_meta = game_info.get('analysis_metadata', {})
        
        for player in merged_data['players']:
            row = {
                'timestamp': datetime.now().isoformat(),
                'quarters_present': ','.join(video_structure.get('quarters_present', [])),
                'total_duration_seconds': video_structure.get('total_duration_seconds', 0),
                'segments_analyzed': ','.join(analysis_meta.get('segments_analyzed', [])),
                'analysis_method': 'two_pass_skim_focus'
            }
            
            detection = player.get('detection_info', {})
            row['player_id'] = player.get('player_id')
            row['team'] = player.get('team')
            row['visible_quarters'] = ','.join(detection.get('visible_quarters', []))
            row['is_starter'] = detection.get('is_starter', False)
            row['estimated_minutes'] = detection.get('estimated_minutes', 0)
            
            detailed = player.get('detailed_analysis', {})
            analyzed_segments = detailed.get('analyzed_segments', [])
            
            total_stats = {
                'fgm': 0, 'fga': 0, 'pts': 0, 'ast': 0, 'reb': 0, 'tov': 0
            }
            
            for idx, segment in enumerate(analyzed_segments, 1):
                seg_stats = segment.get('stats', {})
                pressure = segment.get('pressure_metrics', {})
                temporal = segment.get('temporal_analysis', {})
                
                row[f'seg{idx}_timeframe'] = segment.get('timeframe')
                row[f'seg{idx}_type'] = segment.get('segment_type')
                row[f'seg{idx}_pts'] = seg_stats.get('pts', 0)
                row[f'seg{idx}_fg_pct'] = seg_stats.get('fg_pct', 0)
                row[f'seg{idx}_pressure_intensity'] = pressure.get('pressure_intensity', 0)
                row[f'seg{idx}_mental_toughness'] = pressure.get('mental_toughness', 0)
                row[f'seg{idx}_clutch_factor'] = pressure.get('clutch_factor', '')
                # Ensure temporal exists before accessing its keys
                row[f'seg{idx}_rhythm_confidence'] = temporal.get('rhythm_confidence', '') if temporal else ''
                
                for stat in total_stats.keys():
                    total_stats[stat] += seg_stats.get(stat, 0)
            
            for stat, value in total_stats.items():
                row[f'total_{stat}'] = value
            
            if total_stats['fga'] > 0:
                row['total_fg_pct'] = round((total_stats['fgm'] / total_stats['fga']) * 100, 1)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"✓ Created DataFrame: {len(df)} players × {len(df.columns)} columns")
        return df

    def save_outputs(self, df: pd.DataFrame, merged_data: Dict, base_filename: str):
        """Save CSV and JSON outputs."""
        os.makedirs("output", exist_ok=True)
        
        csv_path = os.path.join("output", f"{base_filename}_twopass.csv")
        json_path = os.path.join("output", f"{base_filename}_twopass.json")
        
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ CSV saved: {csv_path}")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ JSON saved: {json_path}")

    async def run_two_pass_analysis(self, video_path: str):
        """Execute complete two-pass analysis pipeline."""
        logger.info("=" * 60)
        logger.info("TWO-PASS VIDEO ANALYSIS PIPELINE (ENHANCED)")
        logger.info("Approach: Skim → Focus → Merge")
        logger.info("=" * 60)
        
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"Video: {video_path} ({file_size_mb:.2f} MB)\n")
        
        logger.info("Uploading video file (once for both passes)...")
        video_file = None
        try:
            video_file = self.client.files.upload(file=video_path)
            
            while video_file.state.name == "PROCESSING":
                logger.info("Processing video...")
                time.sleep(5)
                video_file = self.client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                logger.error(f"Video processing failed: {video_file.error}")
                return
            logger.info("✓ Video upload complete.")
        
        except Exception as e:
            logger.error(f"Video upload failed: {e}", exc_info=True)
            if video_file:
                self.client.files.delete(name=video_file.name)
            return
        
        try:
            # PASS 1
            skim_result = await self.pass1_skim_analysis(video_file)
            if not skim_result:
                logger.error("Pass 1 failed - aborting")
                return
            
            # PASS 2
            focus_result = await self.pass2_focus_analysis(video_file, skim_result)
            if not focus_result:
                logger.error("Pass 2 failed - aborting")
                return
            
            # MERGE
            merged_result = self.merge_analysis_results(skim_result, focus_result)
            
            # Convert to CSV
            df = self.convert_to_csv(merged_result)
            
            # Generate filename
            video_structure = skim_result.get('video_structure', {})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            team_a = video_structure.get('team_a', 'TeamA').replace(' ', '_')
            team_b = video_structure.get('team_b', 'TeamB').replace(' ', '_')
            base_filename = f"{team_a}_vs_{team_b}_{timestamp}"
            
            self.save_outputs(df, merged_result, base_filename)
            
            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("ANALYSIS COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Method: Two-Pass (Skim + Focus)")
            if not df.empty:
                logger.info(f"Players: {len(df)}")
            logger.info(f"Focus segments analyzed: {len(skim_result.get('focus_segments', []))}")
            logger.info(f"Key moments captured: {len(skim_result.get('key_moments', []))}")
            logger.info("=" * 60)
        
        finally:
            if video_file:
                logger.info(f"Cleaning up uploaded file: {video_file.name}")
                self.client.files.delete(name=video_file.name)
                logger.info("✓ Cleanup complete.")


async def main():
    """Main execution function."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set")
        return
    
    analyzer = TwoPassBasketballAnalyzer(api_key)
    
    video_path = "vid2.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return

    try:
        await analyzer.run_two_pass_analysis(video_path)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    except Exception as e:
        logger.critical(f"\nCritical error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())