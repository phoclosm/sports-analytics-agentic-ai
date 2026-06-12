from google import genai
import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict, List
import logging
import time
import asyncio
from pathlib import Path
import google.api_core.exceptions
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoPassFootballAnalyzer:
    def __init__(self, api_key: str):
        """Initialize analyzer with Gemini API."""
        self.client = genai.Client(api_key=api_key)
        self.skim_model_name = 'models/gemini-2.0-flash-exp'
        self.focus_model_name = 'models/gemini-2.0-flash-exp'
    
    def create_skim_prompt(self) -> str:
        """PASS 1: Lightweight context-gathering prompt for football."""
        return """PASS 1: LIGHTWEIGHT FOOTBALL VIDEO MAPPING
Analyze this football/soccer game video at LOW DETAIL to create a structural map.

YOUR TASK (Fast Scan Only):
1. Identify team names from jerseys/scoreboard
2. Identify which halves/periods are visible in the video
3. Find ALL player substitutions (when players enter/exit)
4. Detect high-pressure moments (close score, penalty situations, final minutes)
5. Identify ALL unique players (jersey numbers/names/positions)
6. Mark EVERY key event timestamp (goals, assists, saves, fouls, cards, corners, penalties)

OUTPUT FORMAT (JSON only, no markdown):
{
  "video_structure": {
    "team_a": "Auto-detected Team A",
    "team_b": "Auto-detected Team B",
    "match_type": "League/Cup/Friendly/etc",
    "total_duration_seconds": 600,
    "periods_present": ["First Half", "Second Half"],
    "game_time_range": "80:00-90:00"
  },
  "players_detected": [
    {
      "player_id": "#10 Messi",
      "team": "Barcelona",
      "position": "Forward",
      "visible_periods": ["Second Half"],
      "is_starter": true,
      "estimated_minutes": 45
    }
  ],
  "key_moments": [
    {
      "timestamp": "85:00-90:00",
      "period": "Second Half",
      "moment_type": "high_pressure",
      "reason": "Tied score, final 5 minutes",
      "score_differential": "0-0"
    },
    {
      "timestamp": "87:30",
      "period": "Second Half",
      "moment_type": "penalty",
      "players_involved": ["#10"],
      "event_details": "Penalty kick awarded"
    }
  ],
  "focus_segments": [
    {
      "start_time": "85:00",
      "end_time": "90:00",
      "period": "Second Half",
      "priority": "HIGH",
      "reason": "Final minutes with tied score - critical pressure"
    }
  ]
}

CRITICAL: Return ONLY valid JSON, no markdown formatting."""

    def create_focus_prompt(self, skim_data: Dict) -> str:
        """PASS 2: Deep analysis prompt for football."""
        focus_segments = skim_data.get('focus_segments', [])
        players = skim_data.get('players_detected', [])
        video_structure = skim_data.get('video_structure', {})
        
        segments_desc = "\n".join([
            f"- {seg['start_time']} to {seg['end_time']} ({seg['period']}): {seg['reason']}"
            for seg in focus_segments
        ])
        
        player_list = ", ".join([p['player_id'] for p in players])
        team_a = video_structure.get('team_a', 'Team A')
        team_b = video_structure.get('team_b', 'Team B')
        
        return f"""PASS 2: FOCUSED DEEP FOOTBALL ANALYSIS
Teams: {team_a} vs {team_b}

FOCUS SEGMENTS:
{segments_desc}

KNOWN PLAYERS: {player_list}

YOUR TASK: Analyze ONLY these segments in MAXIMUM DETAIL for football-specific metrics.

OUTPUT FORMAT (JSON only, no markdown):
{{
  "analysis_metadata": {{
    "segments_analyzed": ["Second Half 85:00-90:00"],
    "total_analysis_duration": "5 minutes",
    "pressure_segments": ["Second Half 85:00-90:00"]
  }},
  "players": [
    {{
      "player_id": "#10 Messi",
      "team": "Barcelona",
      "position": "Forward",
      "role": "Starter",
      "analyzed_segments": [
        {{
          "timeframe": "Second Half 85:00-90:00",
          "segment_type": "high_pressure",
          "minutes_played": 5,
          "stats": {{
            "goals": 1,
            "assists": 0,
            "shots_on_target": 2,
            "shots_off_target": 1,
            "total_shots": 3,
            "shot_accuracy_pct": 66.7,
            "passes_completed": 12,
            "passes_attempted": 15,
            "pass_accuracy_pct": 80.0,
            "key_passes": 2,
            "dribbles_successful": 3,
            "dribbles_attempted": 4,
            "dribble_success_pct": 75.0,
            "tackles_won": 0,
            "interceptions": 0,
            "clearances": 0,
            "fouls_committed": 1,
            "fouls_won": 2,
            "yellow_cards": 0,
            "red_cards": 0
          }},
          "pressure_metrics": {{
            "pressure_intensity": 10,
            "mental_toughness": 9,
            "clutch_factor": "Scored winning goal in 89th minute under intense pressure",
            "critical_plays": 3,
            "decision_making": 9,
            "composure_rating": 9
          }},
          "temporal_analysis": {{
            "rhythm_confidence": "Calm and composed throughout, increased intensity in final 2 minutes",
            "fatigue_indicators": "Slight reduction in sprint speed but maintained technical quality",
            "performance_consistency": "Excellent decision-making under pressure, clutch finish"
          }}
        }}
      ],
      "segment_totals": {{
        "goals": 1,
        "assists": 0,
        "shots_on_target": 2,
        "total_shots": 3,
        "passes_completed": 12,
        "pass_accuracy_pct": 80.0,
        "dribbles_successful": 3,
        "tackles_won": 0
      }}
    }}
  ]
}}

CRITICAL: Return ONLY valid JSON, no markdown formatting, no trailing commas."""

    def clean_json_response(self, text: str) -> str:
        """Clean and extract JSON from API response."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx+1]
        
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        return text

    async def pass1_skim_analysis(self, video_file: genai.types.File) -> Dict:
        """PASS 1: Fast structural analysis with retry logic."""
        logger.info("=" * 60)
        logger.info("PASS 1: SKIM ANALYSIS (Context Gathering)")
        logger.info("=" * 60)
        
        prompt = self.create_skim_prompt()
        logger.info("Running lightweight structural analysis...")
        
        max_retries = 3
        retry_delay_seconds = 5

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.skim_model_name,
                    contents=[video_file, prompt],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        top_p=0.95,
                        response_mime_type="application/json"
                    )
                )
                
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.error("Pass 1: Received empty response from API.")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay_seconds)
                        continue
                    return {}

                cleaned = self.clean_json_response(response.text)
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
                
                return result
                
            except (google.api_core.exceptions.ServiceUnavailable, 
                    google.api_core.exceptions.ResourceExhausted) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Pass 1: Model overloaded (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay_seconds}s...")
                    await asyncio.sleep(retry_delay_seconds)
                    retry_delay_seconds *= 2
                else:
                    logger.error(f"Pass 1 failed after {max_retries} attempts: {e}")
                    return {}
            except json.JSONDecodeError as e:
                logger.error(f"Pass 1 JSON parsing error: {e}")
                logger.error(f"Raw response: {response.text[:500] if response and hasattr(response, 'text') else 'N/A'}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay_seconds)
                    continue
                return {}
            except Exception as e:
                logger.error(f"Pass 1 unexpected error: {e}", exc_info=True)
                return {}
        
        return {}

    async def pass2_focus_analysis(self, video_file: genai.types.File, skim_data: Dict) -> Dict:
        """PASS 2: Deep analysis with enhanced error handling."""
        logger.info("\n" + "=" * 60)
        logger.info("PASS 2: FOCUS ANALYSIS (Deep Dive)")
        logger.info("=" * 60)
        
        focus_segments = skim_data.get('focus_segments', [])
        if not focus_segments:
            logger.warning("No focus segments identified - skipping Pass 2")
            return {}
        
        logger.info(f"Analyzing {len(focus_segments)} high-priority segments")
        
        prompt = self.create_focus_prompt(skim_data)
        
        max_retries = 3
        retry_delay_seconds = 5

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.focus_model_name,
                    contents=[video_file, prompt],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        top_p=0.95,
                        response_mime_type="application/json"
                    )
                )
                
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.error("Pass 2: Received empty response from API.")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay_seconds)
                        continue
                    return {}

                cleaned = self.clean_json_response(response.text)
                result = json.loads(cleaned)
                
                players_analyzed = len(result.get('players', []))
                logger.info(f"✓ Focus analysis complete:")
                logger.info(f"  - {players_analyzed} players analyzed in detail")
                
                return result
                
            except (google.api_core.exceptions.ServiceUnavailable,
                    google.api_core.exceptions.ResourceExhausted) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Pass 2: Model overloaded (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay_seconds}s...")
                    await asyncio.sleep(retry_delay_seconds)
                    retry_delay_seconds *= 2
                else:
                    logger.error(f"Pass 2 failed after {max_retries} attempts: {e}")
                    return {}
            except json.JSONDecodeError as e:
                logger.error(f"Pass 2 JSON parsing error: {e}")
                logger.error(f"Raw response (first 1000 chars): {response.text[:1000] if response and hasattr(response, 'text') else 'N/A'}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay_seconds)
                    continue
                return {}
            except Exception as e:
                logger.error(f"Pass 2 unexpected error: {e}", exc_info=True)
                return {}
        
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
        return merged

    def convert_to_csv(self, merged_data: Dict) -> pd.DataFrame:
        """Convert merged analysis to CSV format."""
        if not merged_data or 'players' not in merged_data:
            return pd.DataFrame()
        
        logger.info("Converting results to CSV...")
        rows = []
        game_info = merged_data.get('game_info', {})
        
        for player in merged_data['players']:
            row = {
                'timestamp': datetime.now().isoformat(),
                'player_id': player.get('player_id'),
                'team': player.get('team')
            }
            
            detection = player.get('detection_info', {})
            row['position'] = detection.get('position', 'Unknown')
            row['is_starter'] = detection.get('is_starter', False)
            row['estimated_minutes'] = detection.get('estimated_minutes', 0)
            
            detailed = player.get('detailed_analysis', {})
            totals = detailed.get('segment_totals', {})
            
            for stat, value in totals.items():
                row[f'total_{stat}'] = value
            
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
        logger.info("TWO-PASS FOOTBALL VIDEO ANALYSIS PIPELINE")
        logger.info("=" * 60)
        
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return
        
        logger.info(f"Video: {video_path}\n")
        logger.info("Uploading video file...")
        
        video_file = None
        try:
            video_file = self.client.files.upload(file=video_path)
            
            while video_file.state.name == "PROCESSING":
                logger.info("Processing video...")
                time.sleep(5)
                video_file = self.client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                logger.error(f"Video processing failed")
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
            
            # Convert and save
            df = self.convert_to_csv(merged_result)
            
            video_structure = skim_result.get('video_structure', {})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            team_a = video_structure.get('team_a', 'TeamA').replace(' ', '_')
            team_b = video_structure.get('team_b', 'TeamB').replace(' ', '_')
            base_filename = f"{team_a}_vs_{team_b}_{timestamp}"
            
            self.save_outputs(df, merged_result, base_filename)
            
            logger.info("\n" + "=" * 60)
            logger.info("FOOTBALL ANALYSIS COMPLETE")
            logger.info("=" * 60)
        
        finally:
            if video_file:
                logger.info(f"Cleaning up uploaded file")
                self.client.files.delete(name=video_file.name)
                logger.info("✓ Cleanup complete.")


async def main():
    """Main execution function."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set")
        return
    
    analyzer = TwoPassFootballAnalyzer(api_key)
    video_path = "football_video.mp4"
    
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