import os
import ray
from data_juicer.utils.constant import Fields
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.mapper.video_calibration_mapper import VideoCalibrationMapper

if __name__ == '__main__':
    # Initialize Ray
    ray.init(address='auto', ignore_reinit_error=True)

    # Setup workspace and paths
    WORKSPACE_DIR = os.getcwd()
    VIDEO_PATH = os.path.join(WORKSPACE_DIR, "test_video.mp4")
    OUTPUT_PATH = os.path.join(WORKSPACE_DIR, 'output/video_calibration/')

    # Create a dummy video for demonstration if it doesn't exist
    import cv2
    import numpy as np
    
    if not os.path.exists(VIDEO_PATH):
        def create_dummy_video(path, frames=30):
            height, width = 480, 640
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, 20.0, (width, height))
            
            for i in range(frames):
                # Create a moving pattern to simulate motion
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.circle(frame, (i * 10 + 50, i * 5 + 50), 20, (255, 255, 255), -1)
                out.write(frame)
            out.release()
            print(f"Created dummy video at {path}")
        
        create_dummy_video(VIDEO_PATH)

    # Create Ray Dataset
    # Data-Juicer uses a list of dictionaries as the basic data structure
    data = [
        {"videos": [VIDEO_PATH], "text": "A dummy video for calibration test."}
    ] * 5 # Duplicate to simulate multiple samples
    
    ray_ds = ray.data.from_items(data)
    ds = RayDataset(ray_ds)

    print(f"Dataset created with {ds.count()} samples.")

    # Initialize the Operator
    # Note: This will automatically download DroidCalib if not present in cache
    op = VideoCalibrationMapper(
        image_size=[384, 512],  # Resize for inference
        stride=2,               # Process every 2nd frame
        max_frames=50,          # Limit frames for speed
        num_gpus=0.5,           # Allocate 0.5 GPU per worker (adjust based on your hardware)
    )

    # Process the Dataset
    # The operator will add 'camera_intrinsics' to the 'meta' field
    ds = ds.process([op])
    
    # Inspect Results
    print("Processing complete. Inspecting results...")
    
    # Convert back to Ray dataset to inspect or write
    processed_ray_ds = ds.data
    
    # Take one sample to show
    samples = processed_ray_ds.take(1)
    if samples:
        sample = samples[0]
        meta = sample.get(Fields.meta, {})
        intrinsics = meta.get("camera_intrinsics")

        if intrinsics is not None:
            print("Estimated Intrinsics (fx, fy, cx, cy):")
            print(intrinsics)
        else:
            print("Calibration failed or no intrinsics found.")
            print("Available meta keys:", list(meta.keys()))

    # Write results
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    processed_ray_ds.write_json(OUTPUT_PATH, force_ascii=False)
    print(f"Results written to {OUTPUT_PATH}")

    # Cleanup dummy video
    if os.path.exists(VIDEO_PATH):
        os.remove(VIDEO_PATH)
