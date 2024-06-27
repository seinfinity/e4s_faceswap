import cv2
import os

def video_to_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    count = 0

    while success:
        frame_path = os.path.join(output_dir, f"frame_{count:04d}.png")
        cv2.imwrite(frame_path, frame)
        success, frame = video_capture.read()
        count += 1

    video_capture.release()
    print(f"Saved {count} frames to {output_dir}")

video_path = "path_to_your_video.mp4"
output_dir = "output_frames"
video_to_frames(video_path, output_dir)
