import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import tempfile
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import time
import threading
import queue
from werkzeug.utils import secure_filename
import subprocess
import gc
import base64

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
FACES_FOLDER = 'detected_faces'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['FACES_FOLDER'] = FACES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

# Global variables
processing_queue = queue.Queue()
processing_status = {}
face_registry = {}
source_faces_registry = {}

# Initialize face app with GPU if available
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cv2.cuda.getCudaEnabledDeviceCount() > 0 else [
    'CPUExecutionProvider']
face_app = FaceAnalysis(name='buffalo_l', providers=providers)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load face swapper model
model_path = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'inswapper_128.onnx')
swapper = get_model(model_path, providers=providers)


class FaceTracker:
    def __init__(self, similarity_threshold=0.90, max_faces=10):
        self.faces = {}
        self.similarity_threshold = similarity_threshold
        self.face_counter = 0
        self.max_faces = max_faces
        self.embeddings = []

    def add_face(self, embedding, thumbnail, bbox, frame_idx):
        # Skip if we already have maximum faces
        if len(self.faces) >= self.max_faces:
            return None

        # Check if this is a new face
        face_id = None
        max_similarity = 0
        best_match_id = None

        for existing_id, existing_face in self.faces.items():
            similarity = np.dot(embedding, existing_face['embedding']) / (
                    np.linalg.norm(embedding) * np.linalg.norm(existing_face['embedding'])
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_id = existing_id

        # Use a higher threshold for matching
        if max_similarity > self.similarity_threshold:
            face_id = best_match_id
            if max_similarity > 0.95:  # Debug very high similarity
                print(f"  High similarity match: {max_similarity:.3f} with {face_id}")

        if face_id is None:
            # New face
            face_id = f"face_{self.face_counter}"
            self.face_counter += 1
            self.faces[face_id] = {
                'embedding': embedding.copy(),
                'thumbnail': thumbnail,
                'bbox': bbox,
                'frames': [frame_idx],
                'best_thumbnail': thumbnail,
                'best_quality': thumbnail.shape[0] * thumbnail.shape[1],
                'appearances': 1,
                'first_seen': frame_idx
            }
            self.embeddings.append(embedding.copy())
            print(f"  New face detected: {face_id}, similarity with others: {max_similarity:.3f}")
        else:
            # Existing face - update with better quality thumbnail
            self.faces[face_id]['frames'].append(frame_idx)
            self.faces[face_id]['appearances'] += 1

            # Update with best quality thumbnail
            current_area = thumbnail.shape[0] * thumbnail.shape[1]
            if current_area > self.faces[face_id]['best_quality']:
                self.faces[face_id]['best_thumbnail'] = thumbnail
                self.faces[face_id]['best_quality'] = current_area
                self.faces[face_id]['embedding'] = embedding.copy()

        return face_id

    def get_faces_data(self, thumbnail_size=128):
        faces_data = []

        # Sort faces by number of appearances (most frequent first)
        sorted_faces = sorted(
            self.faces.items(),
            key=lambda x: len(x[1]['frames']),
            reverse=True
        )

        for face_id, face_data in sorted_faces:
            # Use best thumbnail
            thumbnail = face_data['best_thumbnail']

            # Resize for display
            if thumbnail.shape[0] > thumbnail_size or thumbnail.shape[1] > thumbnail_size:
                scale = thumbnail_size / max(thumbnail.shape[:2])
                new_h = int(thumbnail.shape[0] * scale)
                new_w = int(thumbnail.shape[1] * scale)
                thumbnail = cv2.resize(thumbnail, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to base64
            _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 90])
            thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')

            faces_data.append({
                'id': face_id,
                'thumbnail': f"data:image/jpeg;base64,{thumbnail_base64}",
                'frame_count': len(face_data['frames']),
                'appearances': face_data['appearances'],
                'bbox': face_data['bbox'].tolist(),
                'selected': True
            })

        return faces_data


def detect_faces_in_video_fast(video_path, max_frames=50):
    """Improved face detection with better tracking and deduplication"""
    tracker = FaceTracker(similarity_threshold=0.90, max_faces=20)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate sample rate based on video length
    if total_frames < 300:
        sample_rate = 1
        max_frames_to_process = min(total_frames, 100)
    else:
        sample_rate = max(1, total_frames // max_frames)
        max_frames_to_process = max_frames

    frame_idx = 0
    processed_frames = 0

    print(f"Processing video: {total_frames} frames, sample rate: {sample_rate}")

    # Track previously seen faces in consecutive frames for better matching
    previous_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0 and processed_frames < max_frames_to_process:
            # Process frame
            faces = face_app.get(frame)
            current_frame_faces = []

            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Expand bounding box slightly
                margin = 20
                h, w = frame.shape[:2]
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)

                face_region = frame[y1:y2, x1:x2]
                if face_region.size == 0:
                    continue

                # Try to match with faces from previous frame first
                matched_with_previous = False
                if previous_faces:
                    for prev_face in previous_faces:
                        similarity = np.dot(face.normed_embedding, prev_face['embedding']) / (
                                np.linalg.norm(face.normed_embedding) * np.linalg.norm(prev_face['embedding'])
                        )
                        if similarity > 0.95:  # Very high threshold for consecutive frames
                            # This is likely the same face from previous frame
                            face_id = prev_face['id']
                            tracker.faces[face_id]['frames'].append(frame_idx)
                            tracker.faces[face_id]['appearances'] += 1
                            matched_with_previous = True
                            break

                if not matched_with_previous:
                    # Use tracker to add new or match existing face
                    face_id = tracker.add_face(
                        embedding=face.normed_embedding,
                        thumbnail=face_region,
                        bbox=bbox,
                        frame_idx=frame_idx
                    )

                if face_id:
                    current_frame_faces.append({
                        'id': face_id,
                        'embedding': face.normed_embedding,
                        'bbox': bbox
                    })

            # Update previous faces for next frame
            previous_faces = current_frame_faces
            processed_frames += 1

        frame_idx += 1
        # Stop if we have enough faces or processed enough frames
        if len(tracker.faces) >= tracker.max_faces or frame_idx >= total_frames:
            break

    cap.release()

    # Additional post-processing to merge very similar faces
    # This handles cases where the same face was detected as different in different frames
    faces_to_merge = []
    face_ids = list(tracker.faces.keys())

    for i in range(len(face_ids)):
        for j in range(i + 1, len(face_ids)):
            id1, id2 = face_ids[i], face_ids[j]
            emb1 = tracker.faces[id1]['embedding']
            emb2 = tracker.faces[id2]['embedding']

            similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )

            if similarity > 0.92:  # Merge if very similar
                # Keep the face with more appearances
                if len(tracker.faces[id1]['frames']) >= len(tracker.faces[id2]['frames']):
                    faces_to_merge.append((id2, id1))
                else:
                    faces_to_merge.append((id1, id2))

    # Merge faces
    for from_id, to_id in faces_to_merge:
        if from_id in tracker.faces and to_id in tracker.faces:
            # Merge appearances
            tracker.faces[to_id]['frames'].extend(tracker.faces[from_id]['frames'])
            tracker.faces[to_id]['appearances'] += tracker.faces[from_id]['appearances']

            # Keep best thumbnail
            if tracker.faces[from_id]['best_quality'] > tracker.faces[to_id]['best_quality']:
                tracker.faces[to_id]['best_thumbnail'] = tracker.faces[from_id]['best_thumbnail']
                tracker.faces[to_id]['best_quality'] = tracker.faces[from_id]['best_quality']
                tracker.faces[to_id]['embedding'] = tracker.faces[from_id]['embedding']

            # Remove merged face
            del tracker.faces[from_id]

    print(f"Found {len(tracker.faces)} unique faces after merging")
    for face_id, face_data in tracker.faces.items():
        print(f"  {face_id}: {len(face_data['frames'])} appearances")

    return tracker.get_faces_data()


def process_video_with_mappings(task_id, source_faces, video_path, face_mappings, output_video_path):
    """Process video with face mappings"""
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting face swap...'
        }

        # Prepare mappings
        mapped_faces = []
        for mapping in face_mappings:
            if mapping['selected'] and mapping['source_face_id'] in source_faces:
                mapped_faces.append({
                    'source_face': source_faces[mapping['source_face_id']],
                    'target_face_id': mapping['target_face_id']
                })

        if not mapped_faces:
            raise ValueError("No valid face mappings found")

        # Process video with OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = output_video_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        frame_count = 0
        processing_status[task_id]['progress'] = 5
        processing_status[task_id]['message'] = 'Processing frames...'

        # Create a face tracker for matching during processing
        processing_tracker = FaceTracker(similarity_threshold=0.85)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in this frame
            faces = face_app.get(frame)
            result = frame.copy()

            for face in faces:
                # Get face embedding for matching
                face_embedding = face.normed_embedding

                # Try to match with known target faces from mappings
                matched_mapping = None
                for mapping in mapped_faces:
                    # Get or create face ID for this face
                    bbox = face.bbox.astype(int)
                    face_id = processing_tracker.add_face(
                        face_embedding,
                        result[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox[1] < bbox[3] and bbox[0] < bbox[2] else None,
                        bbox,
                        frame_count
                    )

                    # If this face matches the target face ID, swap it
                    if face_id == mapping['target_face_id']:
                        matched_mapping = mapping
                        break

                # If we found a mapping, swap the face
                if matched_mapping:
                    source_face = matched_mapping['source_face']
                    result = swapper.get(result, face, source_face, paste_back=True)

            out.write(result)
            frame_count += 1

            # Update progress every 10 frames
            if frame_count % 10 == 0:
                progress = min(90, 5 + (frame_count / total_frames) * 85)
                processing_status[task_id]['progress'] = int(progress)
                processing_status[task_id]['message'] = f'Processed {frame_count}/{total_frames} frames'

        cap.release()
        out.release()

        # Re-encode with H.264
        processing_status[task_id]['progress'] = 95
        processing_status[task_id]['message'] = 'Final encoding...'

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', temp_output,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',
            output_video_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            os.remove(temp_output)
        except Exception as e:
            print(f"FFmpeg error: {str(e)}")
            if os.path.exists(temp_output):
                os.rename(temp_output, output_video_path)

        # Cleanup
        try:
            os.remove(video_path)
        except:
            pass

        processing_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Processing complete',
            'output_path': output_video_path
        }

        print(f"[Task {task_id}] Processing complete")

    except Exception as e:
        print(f"[Task {task_id}] Error: {str(e)}")
        processing_status[task_id] = {
            'status': 'error',
            'message': str(e)
        }


def worker():
    """Background worker"""
    while True:
        task = processing_queue.get()
        if task is None:
            break
        task_id, task_data = task
        if task_data['type'] == 'mapped':
            source_faces, video_path, face_mappings, output_path = task_data['data']
            process_video_with_mappings(task_id, source_faces, video_path, face_mappings, output_path)
        processing_queue.task_done()


# Start worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'cuda_available': cv2.cuda.getCudaEnabledDeviceCount() > 0
    })


@app.route('/api/detect-faces', methods=['POST'])
def detect_faces():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save video
        temp_id = str(uuid.uuid4())
        video_filename = f"detect_{temp_id}_{secure_filename(video_file.filename)}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)

        # Detect faces
        print(f"Detecting faces in video: {video_filename}")
        faces_data = detect_faces_in_video_fast(video_path)
        print(f"Found {len(faces_data)} unique faces")

        # Store in registry
        face_registry[temp_id] = {
            'video_path': video_path,
            'faces': faces_data,
            'created_at': time.time()
        }

        return jsonify({
            'detection_id': temp_id,
            'faces': faces_data,
            'face_count': len(faces_data)
        })

    except Exception as e:
        print(f"Error detecting faces: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-source-faces', methods=['POST'])
def upload_source_faces():
    try:
        if 'faces' not in request.files:
            return jsonify({'error': 'No face files'}), 400

        face_files = request.files.getlist('faces')
        task_id = str(uuid.uuid4())
        source_faces = {}

        for i, face_file in enumerate(face_files):
            if face_file.filename == '':
                continue

            # Save face file
            face_id = f"source_{i}"
            face_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{face_id}.jpg")
            face_file.save(face_path)

            # Extract face
            img = cv2.imread(face_path)
            if img is None:
                continue

            faces = face_app.get(img)
            if len(faces) > 0:
                source_faces[face_id] = faces[0]

        if not source_faces:
            return jsonify({'error': 'No valid faces detected'}), 400

        # Store source faces
        source_faces_registry[task_id] = {
            'source_faces': source_faces,
            'created_at': time.time()
        }

        return jsonify({
            'task_id': task_id,
            'face_count': len(source_faces)
        })

    except Exception as e:
        print(f"Error uploading source faces: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/swap-faces-mapped', methods=['POST'])
def swap_faces_mapped():
    try:
        data = request.json
        detection_id = data.get('detection_id')
        source_task_id = data.get('source_task_id')
        face_mappings = data.get('face_mappings', [])

        if not detection_id or not source_task_id:
            return jsonify({'error': 'Missing data'}), 400

        # Get stored data
        detection_data = face_registry.get(detection_id)
        source_data = source_faces_registry.get(source_task_id)

        if not detection_data:
            return jsonify({'error': 'Video data not found'}), 404

        if not source_data:
            return jsonify({'error': 'Source faces not found'}), 404

        # Create task
        task_id = str(uuid.uuid4())
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"output_{task_id}.mp4")

        # Add to queue
        processing_queue.put((task_id, {
            'type': 'mapped',
            'data': (
                source_data['source_faces'],
                detection_data['video_path'],
                face_mappings,
                output_path
            )
        }))

        # Initialize status
        processing_status[task_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Waiting to start...',
            'created_at': time.time()
        }

        return jsonify({
            'task_id': task_id,
            'status_url': f'/api/status/{task_id}',
            'download_url': f'/api/download/{task_id}'
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(processing_status[task_id])


@app.route('/api/download/<task_id>', methods=['GET'])
def download_result(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404

    status = processing_status[task_id]
    if status['status'] != 'completed':
        return jsonify({'error': 'Not ready'}), 400

    output_path = status.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"swapped_{task_id}.mp4"
    )


@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    if task_id in processing_status:
        processing_status[task_id]['status'] = 'cancelled'
        processing_status[task_id]['message'] = 'Cancelled by user'
    return jsonify({'message': 'Cancelled'})


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Clean old files"""
    max_age = 3600
    current_time = time.time()

    # Clean uploads
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if current_time - os.path.getmtime(filepath) > max_age:
            try:
                os.remove(filepath)
            except:
                pass

    # Clean processed files
    for filename in os.listdir(PROCESSED_FOLDER):
        filepath = os.path.join(PROCESSED_FOLDER, filename)
        if current_time - os.path.getmtime(filepath) > max_age:
            try:
                os.remove(filepath)
            except:
                pass

    # Clean registries
    global face_registry, source_faces_registry
    face_registry = {k: v for k, v in face_registry.items()
                     if current_time - v.get('created_at', 0) < max_age}
    source_faces_registry = {k: v for k, v in source_faces_registry.items()
                             if current_time - v.get('created_at', 0) < max_age}

    return jsonify({'message': 'Cleanup complete'})


if __name__ == '__main__':
    print("Face Swapper API Started")
    print(f"CUDA Available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)