import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os
import moviepy as mpy  # Modern MoviePy import


class FaceSwapper:
    def __init__(self, model_path, det_model='buffalo_l'):
        # Initialize face analysis app
        self.app = FaceAnalysis(name=det_model, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load face swapper model
        self.swapper = get_model(model_path, providers=['CPUExecutionProvider'])
        self.source_face = None

    def set_source_face(self, source_image_path):
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            raise ValueError(f"Cannot read image from {source_image_path}")

        source_faces = self.app.get(source_img)
        if len(source_faces) == 0:
            raise ValueError("No faces detected in source image")

        self.source_face = source_faces[0]
        print(f"Source face set successfully.")

    def swap_faces_in_frame(self, frame):
        if self.source_face is None:
            raise ValueError("Source face not set.")

        faces = self.app.get(frame)
        if len(faces) == 0:
            return frame

        result = frame.copy()
        for face in faces:
            result = self.swapper.get(result, face, self.source_face, paste_back=True)
        return result

    def process_video(self, input_video_path, output_video_path):
        print(f"Opening video: {input_video_path}")

        # Load video using MoviePy 2.0 syntax
        video_clip = mpy.VideoFileClip(input_video_path)

        # In MoviePy 2.0, fl_image is replaced or used via transform
        # We define a function that processes the frame
        def frame_processor(get_frame, t):
            frame = get_frame(t)  # Get the frame at time t

            # 1. Convert RGB (MoviePy) to BGR (OpenCV)
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 2. Perform the swap
            swapped_bgr = self.swap_faces_in_frame(bgr_frame)

            # 3. Convert BGR back to RGB for MoviePy
            return cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)

        # Apply transformation (New MoviePy 2.0 method)
        processed_clip = video_clip.transform(frame_processor)

        print("Encoding video for WhatsApp (H.264)...")
        # write_videofile will handle the audio automatically
        processed_clip.write_videofile(
            output_video_path,
            codec="libx264",
            audio_codec="aac",  # Use AAC for best WhatsApp compatibility
            fps=video_clip.fps
        )

        video_clip.close()


def main():
    model_path = r"C:\Users\HP\.insightface\models\inswapper_128.onnx"
    source_face_path = "shamha.jpeg"
    input_video_path = "birthday.mp4"
    output_video_path = "output_video_shamha.mp4"

    face_swapper = FaceSwapper(model_path)
    face_swapper.set_source_face(source_face_path)
    face_swapper.process_video(input_video_path, output_video_path)

    print("\nDone! This file should now be shareable on WhatsApp.")


if __name__ == "__main__":
    main()