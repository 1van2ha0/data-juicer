import os
import sys
import cv2
import numpy as np
import json
import ray
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_split_by_frame_mapper import VideoSplitByFrameMapper
from data_juicer.utils.mm_utils import SpecialTokens

def create_dummy_video(filename, duration_sec=10, fps=30, width=640, height=480):
    """Creates a dummy video with frame counter and timestamp"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    print(f"Creating dummy video: {filename}")
    print(f"Duration: {duration_sec}s, FPS: {fps}, Total Frames: {total_frames}")
    
    for i in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add text info
        text_frame = f"Frame: {i}"
        text_time = f"Time: {i/fps:.2f}s"
        
        # Dynamic effect: a moving circle
        cx = int(width * (i / total_frames))
        cy = height // 2
        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
        
        cv2.putText(frame, text_frame, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, text_time, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"Done. Video saved to {filename}")
    return total_frames

def main():
    # 1. Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    output_dir = os.path.join(current_dir, 'output')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(data_dir, 'dummy_video.mp4')
    jsonl_path = os.path.join(data_dir, 'demo.jsonl')

    # 2. Generate Dummy Video if not exists
    if not os.path.exists(video_path):
        create_dummy_video(video_path, duration_sec=5, fps=30) # 5 seconds = 150 frames

    # 3. Generate JSONL
    # We create a sample with the video
    data = [
        {
            "videos": [video_path],
            "text": f"{SpecialTokens.video} A dummy video for split test. {SpecialTokens.eoc}"
        }
    ]
    
    with open(jsonl_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Demo data prepared at {jsonl_path}")

    # 4. Initialize Data-Juicer Dataset
    # Using NestedDataset (HuggingFace Dataset wrapper) which is the default for local processing
    ds = Dataset.from_json(jsonl_path)
    print(f"Dataset loaded with {len(ds)} samples.")

    # 5. Initialize Operator
    # Split into 50-frame chunks with 10-frame overlap
    # Total 150 frames.
    # Expected splits:
    # 0-50 (len 50)
    # 40-90 (len 50)
    # 80-130 (len 50)
    # 120-150 (len 30) -> Last chunk handling?
    # Let's see the logic.
    # Stride = 50. Overlap = 10.
    # Part 0: Start 0. Len 50+10=60. (0-60)
    # Part 1: Start 50. Len 60. (50-110)
    # Part 2: Start 100. Len 60. (100-160 -> clipped to 150)
    # Part 3: Start 150. (Stop)
    
    op = VideoSplitByFrameMapper(
        split_len=50,
        overlap_len=10,
        unit='frame',
        keep_original_sample=False,
        save_dir=output_dir
    )

    # 6. Process Dataset
    # Note: map() returns a new dataset
    # Since VideoSplitByFrameMapper is a batched operator, we must set batched=True
    processed_ds = ds.map(op.process, batched=True)

    # 7. Show Results
    print("\nProcessing complete. Results:")
    for sample in processed_ds:
        print("-" * 20)
        print(f"Original Text: {sample['text']}")
        print(f"Split Videos: {len(sample['videos'])}")
        for v_path in sample['videos']:
            print(f"  - {os.path.basename(v_path)}")
            
    # 8. Save Results
    # Convert to pandas or list to save as jsonl
    processed_ds.to_json(os.path.join(output_dir, 'result.jsonl'))
    print(f"\nResults saved to {os.path.join(output_dir, 'result.jsonl')}")

if __name__ == '__main__':
    main()
