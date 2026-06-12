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

class TwoPassTennisAnalyzer:
    def __init__(self, api_key: str):
        """Initialize analyzer with Gemini API."""
        self.client = genai.Client(api_key=api_key)
        self.skim_model_name = 'models/gemini-2.0-flash-exp'
        self.focus_model_name = 'models/gemini-2.0-flash-exp'
    
    def create_skim_prompt(self) -> str:
        """PASS 1: Lightweight context-gathering prompt for tennis."""
        return """PASS 1: LIGHTWEIGHT TENNIS VIDEO MAPPING
Analyze this tennis match video at LOW DETAIL to create a structural map.

YOUR TASK (Fast Scan Only):
1. Identify player names from scoreboard/graphics
2. Identify which sets/games are visible in the video
3. Find ALL service games and break points
4. Detect high-pressure moments (tie-breaks, match points, break points, deuce situations)
5. Identify both players and their playing styles
6. Mark EVERY key event timestamp (aces, winners, unforced errors, break points, game points)

OUTPUT FORMAT (JSON only, no markdown):
{
  "video_structure": {
    "player_a": "Auto-detected Player A",
    "player_b": "Auto-detected Player B",
    "match_type": "Grand Slam/ATP/WTA/etc",
    "total_duration_seconds": 600,
    "sets_present": ["Set 3"],
    "game_time_range": "Set 3 Game 8-12"
  },
  "players_detected": [
    {
      "player_id": "Nadal",
      "playing_style": "Baseline grinder",
      "visible_sets": ["Set 3"],
      "dominant_hand": "Left",
      "estimated_games_played": 5
    }
  ],
  "key_moments": [
    {
      "timestamp": "Set 3 5-4",
      "set": "Set 3",
      "moment_type": "break_point",
      "reason": "Break point to serve for the set",
      "score": "5-4, 30-40"
    },
    {
      "timestamp": "Set 3 6-6",
      "set": "Set 3",
      "moment_type": "tiebreak",
      "reason": "Tiebreak to decide set",
      "score": "6-6"
    }
  ],
  "focus_segments": [
    {
      "start_time": "Set 3 5-4",
      "end_time": "Set 3 6-5",
      "set": "Set 3",
      "priority": "HIGH",
      "reason": "Critical break point and service game under pressure"
    }
  ]
}

CRITICAL: Return ONLY valid JSON, no markdown formatting."""

    def create_focus_prompt(self, skim_data: Dict) -> str:
        """PASS 2: Deep analysis prompt for tennis."""
        focus_segments = skim_data.get('focus_segments', [])
        players = skim_data.get('players_detected', [])
        video_structure = skim_data.get('video_structure', {})
        
        segments_desc = "\n".join([
            f"- {seg['start_time']} to {seg['end_time']} ({seg['set']}): {seg['reason']}"
            for seg in focus_segments
        ])
        
        player_list = ", ".join([p['player_id'] for p in players])
        player_a = video_structure.get('player_a', 'Player A')
        player_b = video_structure.get('player_b', 'Player B')
        
        return f"""PASS 2: FOCUSED DEEP TENNIS ANALYSIS
Players: {player_a} vs {player_b}

FOCUS SEGMENTS:
{segments_desc}

KNOWN PLAYERS: {player_list}

YOUR TASK: Analyze ONLY these segments in MAXIMUM DETAIL for tennis-specific metrics.

OUTPUT FORMAT (JSON only, no markdown):
{{
  "analysis_metadata": {{
    "segments_analyzed": ["Set 3 5-4 to 6-5"],
    "total_analysis_duration": "8 minutes",
    "pressure_segments": ["Set 3 5-4"]
  }},
  "players": [
    {{
      "player_id": "Nadal",
      "playing_style": "Baseline grinder",
      "dominant_hand": "Left",
      "role": "Server/Receiver",
      "analyzed_segments": [
        {{
          "timeframe": "Set 3 5-4 to 6-5",
          "segment_type": "high_pressure",
          "games_played": 2,
          "stats": {{
            "aces": 2,
            "double_faults": 1,
            "first_serve_pct": 65.0,
            "first_serve_points_won": 8,
            "first_serve_points_total": 10,
            "second_serve_points_won": 3,
            "second_serve_points_total": 5,
            "service_points_won": 11,
            "service_points_total": 15,
            "service_win_pct": 73.3,
            "break_points_faced": 2,
            "break_points_saved": 1,
            "winners": 5,
            "unforced_errors": 3,
            "winner_error_ratio": 1.67,
            "net_approaches": 2,
            "net_points_won": 1,
            "return_points_won": 4,
            "return_points_total": 12,
            "return_win_pct": 33.3
          }},
          "pressure_metrics": {{
            "pressure_intensity": 10,
            "mental_toughness": 9,
            "clutch_factor": "Saved break point at 30-40 then held serve with ace",
            "critical_plays": 3,
            "decision_making": 9,
            "composure_rating": 9
          }},
          "temporal_analysis": {{
            "rhythm_confidence": "Started nervous but gained confidence after saving break point",
            "fatigue_indicators": "Slight decrease in first serve speed but maintained consistency",
            "performance_consistency": "Excellent clutch serving under extreme pressure"
          }}
        }}
      ],
      "segment_totals": {{
        "aces": 2,
        "double_faults": 1,
        "first_serve_pct": 65.0,
        "service_win_pct": 73.3,
        "winners": 5,
        "unforced_errors": 3,
        "break_points_saved": 1,
        "break_points_faced": 2
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
                logger.info(f"  - Players: {video_structure.get('player_a', 'N/A')} vs {video_structure.get('player_b', 'N/A')}")
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
                'team': 'Individual',
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
        
        for player in merged_data['players']:
            row = {
                'timestamp': datetime.now().isoformat(),
                'player_id': player.get('player_id'),
                'team': player.get('team', 'Individual')
            }
            
            detection = player.get('detection_info', {})
            row['playing_style'] = detection.get('playing_style', 'Unknown')
            row['dominant_hand'] = detection.get('dominant_hand', 'Unknown')
            row['estimated_games_played'] = detection.get('estimated_games_played', 0)
            
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
        logger.info("TWO-PASS TENNIS VIDEO ANALYSIS PIPELINE")
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
            player_a = video_structure.get('player_a', 'PlayerA').replace(' ', '_')
            player_b = video_structure.get('player_b', 'PlayerB').replace(' ', '_')
            base_filename = f"{player_a}_vs_{player_b}_{timestamp}"
            
            self.save_outputs(df, merged_result, base_filename)
            
            logger.info("\n" + "=" * 60)
            logger.info("TENNIS ANALYSIS COMPLETE")
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
    
    analyzer = TwoPassTennisAnalyzer(api_key)
    video_path = "tennis_video.mp4"
    
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