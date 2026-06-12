import typer
import asyncio
import os
import json
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ag import SportsVideoIdentifier
    from basketball_analyzer import TwoPassBasketballAnalyzer
    from n import TwoPassFootballAnalyzer
    from tennis import TwoPassTennisAnalyzer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all analyzer files are in the same directory as main.py.")
    sys.exit(1)

# Initialize Typer application
app = typer.Typer(help="End-to-end Sports Video Analysis Pipeline.")

# Configuration
CLASSIFIER_MODEL = 'sports_classifier1.pkl'
YOLO_MODEL = 'best.pt'
ANALYSIS_OUTPUT_DIR = "output"

async def run_analysis_pipeline(video_path: str, api_key: str):
    """
    Executes the two-stage analysis pipeline:
    1. Sport Classification (agent.py)
    2. Sport-Specific Analysis (basketball/football/tennis_analyzer.py)
    """
    video_file = Path(video_path)
    if not video_file.exists():
        typer.echo(f"❌ Error: Video file not found at '{video_path}'", err=True)
        raise typer.Exit(code=1)

    # Clear previous classification-only files
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    for f in Path(ANALYSIS_OUTPUT_DIR).glob('classification_only_result.json'):
        f.unlink()
        
    typer.echo(f"\n{'='*80}")
    typer.echo(f"🚀 STARTING ANALYSIS for: {video_file.name}")
    typer.echo(f"Input Video Path: {video_path}")
    typer.echo(f"{'='*80}\n")
    
    # STAGE 1: Sport Identification
    try:
        identifier = SportsVideoIdentifier(
            sport_classifier_path=CLASSIFIER_MODEL,
            yolo_model_path=YOLO_MODEL
        )
        
        classification_results = identifier.analyze_video(video_path=video_path, num_frames=5)
        identified_sport = classification_results.get('identified_sport')
        
        typer.echo(f"\n{'='*80}")
        typer.echo(f"CLASSIFICATION RESULT: {identified_sport.upper()}")
        typer.echo(f"{'='*80}\n")

    except Exception as e:
        typer.echo(f"\n❌ STAGE 1 (Classification/Detection) Failed: {e}", err=True)
        raise typer.Exit(code=1)

    # STAGE 2: Sport-Specific Deep Analysis
    if identified_sport == 'basketball':
        typer.echo("\n--- EXECUTING BASKETBALL DEEP ANALYSIS (Gemini API) ---")
        try:
            analyzer = TwoPassBasketballAnalyzer(api_key=api_key)
            await analyzer.run_two_pass_analysis(video_path=video_path)
            typer.echo("\n✅ BASKETBALL ANALYSIS COMPLETE. Results saved to 'output/' directory.")
        except Exception as e:
            typer.echo(f"\n❌ STAGE 2 (Basketball Analysis) Failed: {e}", err=True)
            raise typer.Exit(code=1)
            
    elif identified_sport == 'football':
        typer.echo("\n--- EXECUTING FOOTBALL DEEP ANALYSIS (Gemini API) ---")
        try:
            analyzer = TwoPassFootballAnalyzer(api_key=api_key)
            await analyzer.run_two_pass_analysis(video_path=video_path)
            typer.echo("\n✅ FOOTBALL ANALYSIS COMPLETE. Results saved to 'output/' directory.")
        except Exception as e:
            typer.echo(f"\n❌ STAGE 2 (Football Analysis) Failed: {e}", err=True)
            raise typer.Exit(code=1)
            
    elif identified_sport == 'tennis':
        typer.echo("\n--- EXECUTING TENNIS DEEP ANALYSIS (Gemini API) ---")
        try:
            analyzer = TwoPassTennisAnalyzer(api_key=api_key)
            await analyzer.run_two_pass_analysis(video_path=video_path)
            typer.echo("\n✅ TENNIS ANALYSIS COMPLETE. Results saved to 'output/' directory.")
        except Exception as e:
            typer.echo(f"\n❌ STAGE 2 (Tennis Analysis) Failed: {e}", err=True)
            raise typer.Exit(code=1)
            
    else:
        typer.echo(f"\n⚠️ WARNING: Identified sport is '{identified_sport.upper()}'.")
        typer.echo("Supported sports: basketball, football, tennis")
        typer.echo("Skipping deep analysis. Pipeline completed up to classification stage.")
        
        # Save classification result for dashboard
        with open(Path(ANALYSIS_OUTPUT_DIR) / "classification_only_result.json", 'w') as f:
            json.dump(classification_results, f, indent=2)


@app.command(name="analyze")
def analyze_video_cli(
    video_path: str = typer.Option(
        ..., "--video-path", "-v", 
        help="Path to the video file (e.g., 'vid3.mp4'). [REQUIRED]"
    ),
    api_key: str = typer.Option(
        None, "--api-key", "-k", 
        help="Your Gemini API Key. Can also be set via GEMINI_API_KEY environment variable."
    )
):
    """
    Run the full two-stage sports video analysis pipeline.
    """
    
    # Get API Key
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        typer.echo("❌ Error: GEMINI_API_KEY not set. Please provide it with --api-key or set the environment variable.", err=True)
        raise typer.Exit(code=1)
    
    # Execute Async Pipeline
    try:
        asyncio.run(run_analysis_pipeline(video_path, api_key))
    except KeyboardInterrupt:
        typer.echo("\n\nPipeline interrupted by user.")
        raise typer.Exit()
    except Exception as e:
        typer.echo(f"\n\nCritical Error in Main Pipeline: {e}", err=True)
        raise typer.Exit(code=1)
        
    typer.echo(f"\n{'='*80}")
    typer.echo("✅ PIPELINE FINISHED. Run 'streamlit run dashboard.py' to view results.")
    typer.echo(f"{'='*80}")


if __name__ == "__main__":
    app()