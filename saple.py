import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os



class FaceSwapper:
    def __init__(self,model_path, det_model='buffalo_l'):

        self.app = FaceAnalysis(name=det_model, providers = ['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640,640))

        self.swapper = get_model(model_path,providers= ['CPUExecutionProvider'])

        self.source_face = None


    def set_source_face(self, source_image_path):
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            raise ValueError (f"Cannot read image from {source_image_path}")
        source_faces = self.app.get(source_img)

        if len(source_faces) == 0:
            raise ValueError ("No faces detected in source image")

        self.source_face = source_faces[0]
        print(f"Source face set successfully from {source_image_path}")

    def swap_faces_in_frame(self, frame):
        if self.source_face is None:
            raise ValueError("Source face not set. Call setface function first")
        faces = self.app.get(frame)

        if len(faces) == 0:
            return frame

        result = frame.copy()
        for face in faces:
            result = self.swapper.get(result, face, self.source_face, paste_back = True)

        return result


    def process_video (self, input_video_path, output_video_path, fps=None):
        cap = cv2.VideoCapture(input_video_path)

        frame_width = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = int (cap.get(cv2.CAP_PROP_FPS))

        output_fps = fps if fps is not None else input_fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret :
                    break

                processed_frame = self.swap_faces_in_frame(frame)
                out.write(processed_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames ...")

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}" )

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"Video processing completed. Total frames: {frame_count}")
        print(f"Output saved to: {output_video_path}")

def main():
        # Paths (Update these according to your system)
    model_path = r"C:\Users\HP\.insightface\models\inswapper_128.onnx"
    source_face_path = "shamha.jpeg"
    input_video_path = "birthday.mp4"
    output_video_path = "output_video_shamha.mp4"

        # Initialize face swapper
    print("Initializing Face Swapper...")
    face_swapper = FaceSwapper(model_path, det_model='buffalo_l')

        # Set source face
    print(f"Setting source face from {source_face_path}...")
    face_swapper.set_source_face(source_face_path)

        # Process video
    print(f"Processing video {input_video_path}...")
    face_swapper.process_video(input_video_path, output_video_path)

    print("\nDone!")

    if __name__ == "__main__":
        main()
