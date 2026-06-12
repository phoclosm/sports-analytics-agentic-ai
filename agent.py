import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import sys
from PIL import Image

# --- NEW: Import fastai ---
try:
    from fastai.vision.all import *
    FASTAI_AVAILABLE = True
except ImportError:
    FASTAI_AVAILABLE = False
    print("Warning: fastai not installed. Run: pip install fastai")

# --- YOLO Import ---
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")

@dataclass
class Frame:
    """Represents an extracted video frame"""
    image: np.ndarray
    timestamp: float
    frame_number: int
    detections: List[Dict] = None

@dataclass
class SportScript:
    """Configuration for each sport"""
    name: str
    icon: str
    yolo_classes: List[str]
    rules: List[str]
    tracking: List[str]
    metrics: List[str]

class SportsVideoIdentifier:
    """
    Two-stage sports analysis pipeline:
    Stage 1: Identify sport type from video frames using fastai classifier
    Stage 2: Run sport-specific YOLOv8 object detection
    """

    def __init__(self,
                 sport_classifier_path: Optional[str] = None,
                 yolo_model_path: str = 'best.pt'):
        self.sport_classifier_path = sport_classifier_path
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.sport_classifier = None
        self.classifier_type = None

        # Only load fastai classifier if .pkl file is provided
        if sport_classifier_path and sport_classifier_path.endswith('.pkl'):
            if FASTAI_AVAILABLE:
                try:
                    # Platform-specific patch for loading
                    if sys.platform == "win32":
                        import pathlib
                        temp = pathlib.PosixPath
                        pathlib.PosixPath = pathlib.WindowsPath

                    self.sport_classifier = load_learner(sport_classifier_path)
                    
                    # Restore the original path class
                    if sys.platform == "win32":
                        pathlib.PosixPath = temp
                    
                    self.classifier_type = 'fastai'
                    print(f"✓ Loaded fastai classifier model from: {sport_classifier_path}")
                except Exception as e:
                    print(f"❌ Could not load fastai model: {e}")
                    raise
            else:
                print(f"❌ ERROR: A .pkl model was provided ({sport_classifier_path}), but 'fastai' is not installed.")
                raise ImportError("fastai is required for .pkl models")
        else:
            print(f"❌ ERROR: Only .pkl (fastai) models are supported for classification.")
            print(f"   Please provide a fastai model path ending in .pkl")
            raise ValueError("sport_classifier_path must be a .pkl file")

        # Load YOLO model for object detection only
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                print(f"✓ Loaded YOLOv8 object detection model from: {yolo_model_path}")
            except Exception as e:
                print(f"⚠️ Could not load YOLOv8 model: {e}")
        else:
            print("⚠️ YOLO not available for object detection")

        self.sports_scripts = {
            'basketball': SportScript(
                name='Basketball', icon='🏀', yolo_classes=['BasketPlayer', 'Basketball', 'Hoop'],
                rules=['5 players per team', 'Score by shooting through hoop', '4 quarters of 12 minutes'],
                tracking=['Player positions', 'Ball trajectory', 'Shot accuracy', 'Hoop detection'],
                metrics=['Field goal %', 'Three-point %', 'Rebounds', 'Assists']
            ),
            'football': SportScript(
                name='Football', icon='⚽', yolo_classes=['FootPlayer', 'Football', 'goalkeeper'],
                rules=['11 players per team', 'Score by kicking ball into goal', '2 halves of 45 minutes'],
                tracking=['Player formations', 'Ball possession', 'Goalkeeper actions', 'Shot locations'],
                metrics=['Possession %', 'Pass accuracy', 'Shots on target', 'Goalkeeper saves']
            ),
            'tennis': SportScript(
                name='Tennis', icon='🎾', yolo_classes=['tennis ball', 'tennis racket', 'person'],
                rules=['Singles or doubles', 'Points, games, and sets', 'Serve must land in service box'],
                tracking=['Ball speed and trajectory', 'Player court position', 'Shot placement', 'Rally length'],
                metrics=['Serve accuracy', 'Aces', 'Unforced errors', 'Winners']
            )
        }

    def extract_key_frames(self, video_path: str, num_frames: int = 5) -> List[Frame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 Video Info: {total_frames} frames, {fps:.2f} FPS")
        
        # Evenly space frames from 10% to 90% of the video
        frame_positions = np.linspace(0.1, 0.9, num_frames)
        
        extracted_frames = []
        print(f"\n🎞️ Extracting {num_frames} key frames...")
        for position in frame_positions:
            frame_num = int(total_frames * position)
            timestamp = frame_num / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                extracted_frames.append(Frame(image=frame, timestamp=timestamp, frame_number=frame_num, detections=[]))
                print(f"  ✓ Frame at {timestamp:.2f}s (frame {frame_num}, position {position*100:.0f}%)")
            else:
                print(f"  ✗ Failed to extract frame at position {position}")
        cap.release()
        return extracted_frames

    def identify_sport_stage1(self, frames: List[Frame]) -> Tuple[str, Dict]:
        """
        Identify sport using fastai classifier only
        """
        print(f"\n🎯 STAGE 1: Identifying sport type...")
        print(f"  Analyzing {len(frames)} frames for sport classification...")
        
        if self.classifier_type != 'fastai':
            raise ValueError("Only fastai classification is supported")
        
        return self._classify_with_fastai(frames)

    def _classify_with_fastai(self, frames: List[Frame]) -> Tuple[str, Dict]:
        """
        Classify sport using fastai model - Direct inference method for Windows compatibility
        """
        print("  Using fastai ResNet classifier model...")
        import torch
        from torchvision import transforms
        
        sport_names = list(self.sport_classifier.dls.vocab)
        sport_votes = []
        all_probs = []
        
        # Get the model and put it in eval mode
        model = self.sport_classifier.model
        model.eval()
        
        # Define preprocessing transforms (match what fastai uses)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for i, frame in enumerate(frames):
                # Convert BGR (OpenCV) to RGB
                rgb_frame = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # Preprocess the image
                input_tensor = preprocess(pil_image)
                input_batch = input_tensor.unsqueeze(0).to(device)
                
                # Get model prediction
                output = model(input_batch)
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(output, dim=1)
                probs_np = probs.cpu().numpy()[0]
                
                # Get predicted class
                pred_idx = int(probs_np.argmax())
                predicted_sport = sport_names[pred_idx]
                confidence = float(probs_np[pred_idx])
                
                sport_votes.append(predicted_sport)
                all_probs.append({name: float(p) for name, p in zip(sport_names, probs_np)})
                print(f"      Frame {i+1}: {predicted_sport} ({confidence*100:.2f}%)")
        
        # Majority voting
        vote_counts = Counter(sport_votes)
        identified_sport = vote_counts.most_common(1)[0][0]
        
        # Average probabilities across all frames
        predictions = {sport: 0.0 for sport in sport_names}
        for sport in sport_names:
            sport_confs = [p.get(sport, 0) for p in all_probs]
            predictions[sport] = sum(sport_confs) / len(frames) if frames else 0.0
        
        print(f"\n✨ Sport Classification Results:")
        for sport, conf in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            icon = self.sports_scripts.get(sport, SportScript('', '❓', [], [], [], [])).icon
            print(f"  {icon} {sport.upper()}: {conf*100:.2f}% confidence")
        print(f"  🗳️  Majority vote: {identified_sport} ({vote_counts[identified_sport]}/{len(frames)} frames)")
        
        return identified_sport, predictions

    def detect_objects_stage2(self, frames: List[Frame], sport: str, confidence: float = 0.25) -> List[Frame]:
        if not self.yolo_model:
            print("⚠️ YOLO model not available for object detection")
            return frames
            
        script = self.sports_scripts[sport]
        print(f"\n🔍 STAGE 2: Running YOLOv8 object detection for {script.name}...")
        print(f"  Target classes: {', '.join(script.yolo_classes)}")
        
        for i, frame in enumerate(frames):
            print(f"  Analyzing frame {i+1}/{len(frames)}...")
            results = self.yolo_model(frame.image, conf=confidence, verbose=False)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]
                    if class_name in script.yolo_classes:
                        detections.append({
                            'class': class_name,
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        })
            
            frame.detections = detections
            if detections:
                det_summary = Counter([d['class'] for d in detections])
                print(f"      Found {len(detections)} objects: {dict(det_summary)}")
            else:
                print(f"      No relevant objects detected")
        
        return frames

    def save_frames_with_detections(self, frames: List[Frame], sport: str, output_dir: str = "output_frames"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        script = self.sports_scripts[sport]
        
        print(f"\n💾 Saving annotated frames to {output_dir}/...")
        for i, frame in enumerate(frames):
            annotated_frame = frame.image.copy()
            cv2.putText(annotated_frame, f"{script.icon} {script.name}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            if frame.detections:
                for det in frame.detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det['class']} {det['confidence']:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            filename = f"{sport}_frame_{i+1}_t{frame.timestamp:.2f}s.jpg"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), annotated_frame)
            print(f"  ✓ {filename}")

    def execute_sport_script(self, sport: str, video_path: str, frames: List[Frame]):
        if sport not in self.sports_scripts:
            return
            
        script = self.sports_scripts[sport]
        print(f"\n{'='*70}\n🎯 EXECUTING {script.name.upper()} ANALYSIS SCRIPT\n{'='*70}")
        print(f"\n{script.icon} Sport: {script.name}")
        print(f"📹 Video: {Path(video_path).name}")
        
        all_detections = [d['class'] for frame in frames if frame.detections for d in frame.detections]
        if all_detections:
            print(f"\n🔍 Detected Objects:")
            for obj, count in Counter(all_detections).most_common():
                print(f"  • {obj}: {count} detections")
        
        print(f"\n📋 Game Rules:")
        [print(f"  • {rule}") for rule in script.rules]
        
        print(f"\n🎯 Tracking Parameters:")
        [print(f"  • {param}") for param in script.tracking]
        
        print(f"\n📊 Performance Metrics:")
        [print(f"  • {metric}") for metric in script.metrics]
        
        print(f"\n{'='*70}\n✅ Script execution complete!\n{'='*70}\n")

    def analyze_video(self, video_path: str, num_frames: int = 5, 
                     save_frames_flag: bool = True, confidence: float = 0.25) -> Dict:
        print(f"\n{'='*70}\n🎬 TWO-STAGE SPORTS VIDEO ANALYSIS PIPELINE\n{'='*70}\n")
        
        print("📍 Step 1: Extracting key frames...")
        frames = self.extract_key_frames(video_path, num_frames)
        
        identified_sport, sport_confidence = self.identify_sport_stage1(frames)
        
        frames = self.detect_objects_stage2(frames, identified_sport, confidence)
        
        if save_frames_flag:
            print("\n📍 Step 3: Saving annotated frames...")
            self.save_frames_with_detections(frames, identified_sport)
        
        print("\n📍 Step 4: Executing sport-specific analysis...")
        self.execute_sport_script(identified_sport, video_path, frames)
        
        results = {
            'video_path': video_path,
            'identified_sport': identified_sport,
            'sport_confidence': sport_confidence,
            'num_frames_analyzed': len(frames),
            'total_detections': sum(len(f.detections) for f in frames if f.detections),
            'sport_script': asdict(self.sports_scripts[identified_sport])
        }
        
        return results


if __name__ == "__main__":
    # Only use fastai classifier
    identifier = SportsVideoIdentifier(
        sport_classifier_path='sports_classifier.pkl',
        yolo_model_path='best.pt'
    )
    
    video_path = "C:/wierdapproach/vid3.mp4"  # Update with your video path

    try:
        results = identifier.analyze_video(
            video_path=video_path,
            num_frames=5, 
            save_frames_flag=True,
            confidence=0.25
        )

        if results:
            with open('analysis_results.json', 'w') as f:
                serializable_results = {
                    'video_path': results['video_path'],
                    'identified_sport': results['identified_sport'],
                    'sport_confidence': results['sport_confidence'],
                    'num_frames_analyzed': results['num_frames_analyzed'],
                    'total_detections': results['total_detections'],
                    'sport_name': results['sport_script']['name']
                }
                json.dump(serializable_results, f, indent=2)
            
            print("\n✅ Analysis complete!")
            print(f"  Results saved to: analysis_results.json")
            print(f"  Annotated frames saved to: output_frames/")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()