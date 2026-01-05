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
import traceback

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


# Enhanced FaceTracker class with better tracking
class FaceTracker:
    def __init__(self, similarity_threshold=0.6, min_face_size=40):
        self.faces = {}
        self.face_counter = 0
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.max_faces_to_track = 20
        self.fade_frames = 10  # Number of frames a face can be missing before being removed

    def update(self, frame, frame_num):
        """Update face tracker with new frame"""
        # Detect faces in current frame
        detected_faces = face_app.get(frame)

        # Create array for detected face embeddings
        current_embeddings = []
        for face in detected_faces:
            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]

            # Skip faces that are too small
            if face_width < self.min_face_size or face_height < self.min_face_size:
                continue

            current_embeddings.append({
                'face': face,
                'embedding': face.normed_embedding,
                'bbox': bbox
            })

        # Match detected faces with tracked faces
        matched_indices = set()

        for track_id, track_data in self.faces.items():
            if track_data['active']:
                best_match_idx = -1
                best_similarity = self.similarity_threshold

                for i, det_face in enumerate(current_embeddings):
                    if i in matched_indices:
                        continue

                    # Calculate cosine similarity
                    similarity = self.cosine_similarity(
                        track_data['embedding'],
                        det_face['embedding']
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_idx = i

                if best_match_idx >= 0:
                    # Update tracked face
                    det_face = current_embeddings[best_match_idx]
                    self.faces[track_id].update({
                        'embedding': det_face['embedding'],
                        'bbox': det_face['bbox'],
                        'last_seen': frame_num,
                        'frame_count': track_data['frame_count'] + 1,
                        'active': True
                    })

                    # Update face object reference
                    self.faces[track_id]['face'] = det_face['face']

                    matched_indices.add(best_match_idx)
                else:
                    # Face not detected in this frame
                    frames_missing = frame_num - track_data['last_seen']
                    if frames_missing > self.fade_frames:
                        self.faces[track_id]['active'] = False
                    else:
                        self.faces[track_id]['active'] = True

        # Add new faces for unmatched detections
        for i, det_face in enumerate(current_embeddings):
            if i not in matched_indices:
                # Limit number of tracked faces
                if len(self.faces) < self.max_faces_to_track:
                    face_id = f"face_{self.face_counter}"
                    self.face_counter += 1

                    self.faces[face_id] = {
                        'face': det_face['face'],
                        'embedding': det_face['embedding'],
                        'bbox': det_face['bbox'],
                        'first_seen': frame_num,
                        'last_seen': frame_num,
                        'frame_count': 1,
                        'active': True
                    }

        # Clean up inactive faces
        faces_to_remove = []
        for track_id, track_data in self.faces.items():
            if not track_data['active']:
                faces_to_remove.append(track_id)

        for track_id in faces_to_remove:
            del self.faces[track_id]

    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1_norm, emb2_norm)

    def get_active_faces(self):
        """Get all active tracked faces"""
        active_faces = {}
        for track_id, track_data in self.faces.items():
            if track_data['active']:
                active_faces[track_id] = track_data
        return active_faces


def detect_faces_in_video_fast(video_path, max_samples=50):
    """Detect unique faces in video - each face only once with best quality thumbnail"""
    print(f"Starting face detection for: {video_path}")

    tracker = FaceTracker(similarity_threshold=0.85, min_face_size=40)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video info: {total_frames} frames, {fps} fps")

    # Determine sampling strategy
    if total_frames <= max_samples:
        # Process all frames for short videos
        sample_indices = list(range(total_frames))
    else:
        # Sample frames evenly throughout the video
        step = total_frames // max_samples
        sample_indices = list(range(0, total_frames, step))[:max_samples]

    print(f"Will sample {len(sample_indices)} frames for face detection")

    processed_samples = 0
    best_thumbnails = {}  # Store best thumbnail for each face

    for frame_idx in sample_indices:
        # Seek to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Update face tracker with this frame
        tracker.update(frame, frame_idx)
        processed_samples += 1

        # Store best thumbnails for active faces
        active_faces = tracker.get_active_faces()
        for face_id, face_data in active_faces.items():
            bbox = face_data['bbox']
            x1, y1, x2, y2 = bbox

            # Extract face thumbnail
            margin = 10
            h, w = frame.shape[:2]
            x1_exp = max(0, x1 - margin)
            y1_exp = max(0, y1 - margin)
            x2_exp = min(w, x2 + margin)
            y2_exp = min(h, y2 + margin)

            face_region = frame[y1_exp:y2_exp, x1_exp:x2_exp]

            if face_region.size == 0:
                continue

            # Calculate quality score
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height

            if face_id not in best_thumbnails:
                best_thumbnails[face_id] = {
                    'thumbnail': face_region,
                    'quality_score': face_area,
                    'embedding': face_data['embedding'],
                    'bbox': bbox,
                    'frame_count': 1
                }
            else:
                # Keep best quality thumbnail
                if face_area > best_thumbnails[face_id]['quality_score']:
                    best_thumbnails[face_id] = {
                        'thumbnail': face_region,
                        'quality_score': face_area,
                        'embedding': face_data['embedding'],
                        'bbox': bbox,
                        'frame_count': best_thumbnails[face_id]['frame_count'] + 1
                    }
                else:
                    best_thumbnails[face_id]['frame_count'] += 1

        # Print progress
        if processed_samples % 10 == 0:
            print(
                f"  Processed {processed_samples}/{len(sample_indices)} samples, found {len(best_thumbnails)} unique faces")

    cap.release()

    print(f"\nDetection complete:")
    print(f"  Processed frames: {processed_samples}")
    print(f"  Unique faces found: {len(best_thumbnails)}")

    # Prepare faces data with embeddings
    faces_data = []
    thumbnail_size = 160

    for face_id, face_data in best_thumbnails.items():
        # Skip faces with very few detections (likely false positives)
        if face_data['frame_count'] < 2:
            continue

        thumbnail = face_data['thumbnail']

        # Resize thumbnail for consistent display
        h, w = thumbnail.shape[:2]
        if h > thumbnail_size or w > thumbnail_size:
            scale = thumbnail_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            thumbnail = cv2.resize(thumbnail, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 90])
        thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')

        # Store embedding as list for JSON serialization
        embedding_list = face_data['embedding'].tolist()

        faces_data.append({
            'id': face_id,
            'thumbnail': f"data:image/jpeg;base64,{thumbnail_base64}",
            'frame_count': face_data['frame_count'],
            'appearances': face_data['frame_count'],
            'bbox': face_data['bbox'].tolist(),
            'selected': True,
            'embedding': embedding_list,  # Store embedding for matching
            'quality_score': float(face_data['quality_score'])
        })

    print(f"Returning {len(faces_data)} unique faces with embeddings")
    return faces_data


def process_video_with_mappings(task_id, detection_id, source_faces, video_path, face_mappings, output_video_path):
    """Process video with face mappings - tracks faces throughout entire video"""
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting face swap...'
        }

        # Get target face embeddings from the detection phase
        if detection_id not in face_registry:
            raise ValueError(f"Detection data not found for ID: {detection_id}")

        detection_data = face_registry[detection_id]
        detected_faces = detection_data.get('faces', [])

        # Build target face database from detection
        target_faces_db = {}
        for face_data in detected_faces:
            face_id = face_data['id']
            if 'embedding' in face_data:
                # Convert list back to numpy array
                target_faces_db[face_id] = {
                    'embedding': np.array(face_data['embedding'], dtype=np.float32),
                    'source_face_id': None
                }

        # Map target faces to source faces based on user mappings
        for mapping in face_mappings:
            if mapping['selected'] and mapping['source_face_id'] in source_faces:
                target_face_id = mapping['target_face_id']
                if target_face_id in target_faces_db:
                    target_faces_db[target_face_id]['source_face_id'] = mapping['source_face_id']
                    target_faces_db[target_face_id]['source_face'] = source_faces[mapping['source_face_id']]

        # Check if we have valid mappings
        valid_mappings = {k: v for k, v in target_faces_db.items()
                          if v.get('source_face_id') is not None}

        if not valid_mappings:
            raise ValueError("No valid face mappings found")

        print(f"[Task {task_id}] Starting face swap with {len(valid_mappings)} mappings")

        # Read video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            # Try alternative method to get frame count
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            total_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print(f"[Task {task_id}] Video info: {width}x{height}, {fps}fps, {total_frames} frames")

        # Create output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = output_video_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        frame_count = 0
        processing_status[task_id]['progress'] = 5
        processing_status[task_id]['message'] = 'Initializing face tracker...'

        # Initialize face tracker for processing
        processor_tracker = FaceTracker(similarity_threshold=0.7, min_face_size=40)

        # Store which tracked faces have been mapped to which source faces
        face_mapping_cache = {}  # processor_face_id -> source_face

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            result = frame.copy()

            # Update face tracker with current frame
            processor_tracker.update(frame, frame_count)

            # Get active faces in current frame
            active_faces = processor_tracker.get_active_faces()

            # Process each active face
            for processor_face_id, face_data in active_faces.items():
                face_obj = face_data['face']
                face_embedding = face_data['embedding']

                # Check if we already mapped this processor face
                if processor_face_id in face_mapping_cache:
                    # We already know which source face to use
                    source_face = face_mapping_cache[processor_face_id]
                    result = swapper.get(result, face_obj, source_face, paste_back=True)
                else:
                    # First time seeing this processor face, find best match with target faces
                    best_match_id = None
                    best_similarity = 0.6  # Matching threshold

                    for target_face_id, target_data in valid_mappings.items():
                        target_embedding = target_data['embedding']

                        # Calculate similarity
                        similarity = processor_tracker.cosine_similarity(
                            face_embedding,
                            target_embedding
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_id = target_face_id

                    # If we found a good match, use it and cache the mapping
                    if best_match_id:
                        source_face = valid_mappings[best_match_id]['source_face']
                        face_mapping_cache[processor_face_id] = source_face
                        result = swapper.get(result, face_obj, source_face, paste_back=True)

            # Write processed frame
            out.write(result)

            # Update progress
            if total_frames > 0 and frame_count % 10 == 0:
                progress = min(90, 5 + (frame_count / total_frames) * 85)
                processing_status[task_id]['progress'] = int(progress)
                processing_status[task_id][
                    'message'] = f'Processed {frame_count}/{total_frames} frames ({len(face_mapping_cache)} faces mapped)'

        cap.release()
        out.release()

        print(f"[Task {task_id}] Video processing complete. Mapped {len(face_mapping_cache)} unique faces")

        # Re-encode with H.264
        processing_status[task_id]['progress'] = 95
        processing_status[task_id]['message'] = 'Final encoding...'

        if os.path.exists(output_video_path):
            os.remove(output_video_path)

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', temp_output,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_video_path
        ]

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            os.remove(temp_output)
            print(f"[Task {task_id}] FFmpeg encoding successful")
        except subprocess.CalledProcessError as e:
            print(f"[Task {task_id}] FFmpeg error: {e.stderr}")
            # If FFmpeg fails, try to use the temp file
            if os.path.exists(temp_output):
                os.rename(temp_output, output_video_path)
        except Exception as e:
            print(f"[Task {task_id}] FFmpeg exception: {str(e)}")
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
            'output_path': output_video_path,
            'faces_mapped': len(face_mapping_cache)
        }

        print(f"[Task {task_id}] Processing complete - output: {output_video_path}")

    except Exception as e:
        print(f"[Task {task_id}] Error: {str(e)}")
        traceback.print_exc()
        processing_status[task_id] = {
            'status': 'error',
            'message': str(e)
        }

        # Cleanup on error
        try:
            cap.release()
        except:
            pass
        try:
            out.release()
        except:
            pass


def worker():
    """Background worker"""
    while True:
        task = processing_queue.get()
        if task is None:
            break
        task_id, task_data = task
        if task_data['type'] == 'mapped':
            detection_id, source_faces, video_path, face_mappings, output_path = task_data['data']
            process_video_with_mappings(task_id, detection_id, source_faces, video_path, face_mappings, output_path)
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
        print(f"\n=== Detecting faces in video: {video_filename} ===")
        faces_data = detect_faces_in_video_fast(video_path)
        print(f"=== Found {len(faces_data)} unique faces ===\n")

        # Store in registry WITH EMBEDDINGS
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
        traceback.print_exc()
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

            # Clean up temp file
            try:
                os.remove(face_path)
            except:
                pass

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
        traceback.print_exc()
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
                detection_id,  # Pass detection_id to get embeddings
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
        traceback.print_exc()
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
    max_age = 3600  # 1 hour
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

    # Clean processing status
    old_tasks = []
    for task_id, status in processing_status.items():
        if 'created_at' in status and current_time - status['created_at'] > max_age * 2:
            old_tasks.append(task_id)

    for task_id in old_tasks:
        del processing_status[task_id]

    return jsonify({'message': 'Cleanup complete', 'files_cleaned': len(old_tasks)})


if __name__ == '__main__':
    print("Face Swapper API Started")
    print(f"CUDA Available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Processed folder: {PROCESSED_FOLDER}")

    # Run cleanup on start
    try:
        cleanup()
    except:
        pass

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)