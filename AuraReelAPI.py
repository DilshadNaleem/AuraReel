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
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global variables for tracking processing
processing_queue = queue.Queue()
processing_status = {}


class FaceSwapperAPI:
    def __init__(self, model_path):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = get_model(model_path, providers=['CPUExecutionProvider'])
        print("Face swapper model loaded successfully")

    def set_source_face(self, source_image_path):
        """Extract face from source image"""
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            raise ValueError(f"Cannot read image from {source_image_path}")

        source_faces = self.app.get(source_img)
        if len(source_faces) == 0:
            raise ValueError("No faces detected in source image")

        return source_faces[0]

    def swap_faces_in_frame(self, frame, source_face):
        """Swap faces in a single frame"""
        faces = self.app.get(frame)
        if len(faces) == 0:
            return frame

        result = frame.copy()
        for face in faces:
            result = self.swapper.get(result, face, source_face, paste_back=True)
        return result

    def process_video_task(self, task_id, source_image_path, input_video_path, output_video_path):
        """Process video using OpenCV"""
        try:
            processing_status[task_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Extracting source face...'
            }

            # Load source face
            source_face = self.set_source_face(source_image_path)
            processing_status[task_id]['progress'] = 10
            processing_status[task_id]['message'] = 'Opening video file...'

            # Open video using OpenCV
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {input_video_path}")

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            processing_status[task_id]['progress'] = 20
            processing_status[task_id]['message'] = f'Processing video: 0/{total_frames} frames'

            frame_count = 0
            processed_frames = 0

            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Swap faces in frame
                processed_frame = self.swap_faces_in_frame(frame, source_face)

                # Write processed frame
                out.write(processed_frame)

                frame_count += 1
                processed_frames += 1

                # Update progress every 10 frames or 10%
                if frame_count % max(1, total_frames // 10) == 0 or frame_count == total_frames:
                    progress = min(80, 20 + (frame_count / total_frames) * 60)
                    processing_status[task_id]['progress'] = int(progress)
                    processing_status[task_id]['message'] = f'Processing video: {frame_count}/{total_frames} frames'

            # Release video capture and writer
            cap.release()
            out.release()

            processing_status[task_id]['progress'] = 90
            processing_status[task_id]['message'] = 'Finalizing output...'

            # Convert from MP4V to H.264 for better compatibility
            # (Optional: Use ffmpeg if available, otherwise keep as is)
            h264_output_path = output_video_path.replace('.mp4', '_h264.mp4')
            try:
                # Try to use ffmpeg if available
                import subprocess
                cmd = [
                    'ffmpeg', '-y', '-i', output_video_path,
                    '-c:v', 'libx264', '-preset', 'medium',
                    '-crf', '23', '-c:a', 'aac', '-b:a', '128k',
                    h264_output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # Remove old file and rename new one
                os.remove(output_video_path)
                os.rename(h264_output_path, output_video_path)
            except:
                # If ffmpeg is not available, keep the original MP4V file
                print("FFmpeg not available, keeping MP4V format")

            # Clean up input files
            if os.path.exists(source_image_path):
                os.remove(source_image_path)
            if os.path.exists(input_video_path):
                os.remove(input_video_path)

            processing_status[task_id] = {
                'status': 'completed',
                'progress': 100,
                'message': 'Processing complete',
                'output_path': output_video_path,
                'duration': frame_count / fps if fps > 0 else 0,
                'fps': fps,
                'total_frames': total_frames
            }

        except Exception as e:
            print(f"Error in task {task_id}: {str(e)}")
            processing_status[task_id] = {
                'status': 'error',
                'progress': 0,
                'message': str(e)
            }
            # Clean up on error
            for path in [source_image_path, input_video_path, output_video_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except:
                    pass


def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False

    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False


def worker():
    """Background worker to process videos"""
    while True:
        task = processing_queue.get()
        if task is None:
            break
        task_id, source_path, video_path, output_path = task
        face_swapper.process_video_task(task_id, source_path, video_path, output_path)
        processing_queue.task_done()


# Initialize face swapper
model_path = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'inswapper_128.onnx')
face_swapper = FaceSwapperAPI(model_path)

# Start background worker
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Face Swapper API is running',
        'model_loaded': True
    })


@app.route('/api/swap-faces', methods=['POST'])
def swap_faces():
    """Main endpoint to start face swapping"""
    try:
        # Check if files are provided
        if 'source_image' not in request.files:
            return jsonify({'error': 'Missing source_image file'}), 400
        if 'target_video' not in request.files:
            return jsonify({'error': 'Missing target_video file'}), 400

        source_file = request.files['source_image']
        video_file = request.files['target_video']

        # Check if files are selected
        if source_file.filename == '':
            return jsonify({'error': 'No source image selected'}), 400
        if video_file.filename == '':
            return jsonify({'error': 'No target video selected'}), 400

        # Validate file types
        if not allowed_file(source_file.filename, 'image'):
            return jsonify({'error': 'Invalid image format. Use JPG, PNG, or BMP'}), 400

        if not allowed_file(video_file.filename, 'video'):
            return jsonify({'error': 'Invalid video format. Use MP4, MOV, AVI, or MKV'}), 400

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Save uploaded files with secure filenames
        source_filename = secure_filename(f"source_{task_id}_{source_file.filename}")
        video_filename = secure_filename(f"video_{task_id}_{video_file.filename}")
        output_filename = f"output_{task_id}.mp4"

        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

        # Save files
        source_file.save(source_path)
        video_file.save(video_path)

        # Validate that files were saved
        if not os.path.exists(source_path):
            return jsonify({'error': 'Failed to save source image'}), 500
        if not os.path.exists(video_path):
            return jsonify({'error': 'Failed to save video'}), 500

        # Add task to queue
        processing_queue.put((task_id, source_path, video_path, output_path))

        # Initialize task status
        processing_status[task_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Waiting to start processing...',
            'created_at': time.time()
        }

        return jsonify({
            'task_id': task_id,
            'message': 'Processing started',
            'status_url': f'/api/status/{task_id}',
            'download_url': f'/api/download/{task_id}'
        }), 202

    except Exception as e:
        print(f"Error in swap_faces endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get processing status for a task"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404

    status_info = processing_status[task_id].copy()

    # Add estimated time remaining for processing tasks
    if status_info['status'] == 'processing':
        progress = status_info['progress']
        if progress > 0 and progress < 100:
            # Estimate based on progress
            elapsed = time.time() - status_info.get('created_at', time.time())
            estimated_total = elapsed / (progress / 100) if progress > 0 else 0
            estimated_remaining = estimated_total - elapsed
            status_info['estimated_seconds_remaining'] = max(0, int(estimated_remaining))

    return jsonify(status_info)


@app.route('/api/download/<task_id>', methods=['GET'])
def download_result(task_id):
    """Download the processed video"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404

    status_info = processing_status[task_id]

    if status_info['status'] != 'completed':
        return jsonify({'error': 'Video not ready yet'}), 400

    output_path = status_info.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404

    return send_file(
        output_path,
        as_attachment=True,
        download_name=f'face_swapped_{task_id}.mp4',
        mimetype='video/mp4'
    )


@app.route('/api/preview/<task_id>', methods=['GET'])
def preview_result(task_id):
    """Get a preview frame from the processed video"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404

    status_info = processing_status[task_id]

    if status_info['status'] != 'completed':
        return jsonify({'error': 'Video not ready yet'}), 400

    output_path = status_info.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404

    # Extract first frame for preview
    cap = cv2.VideoCapture(output_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        from flask import Response
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    return jsonify({'error': 'Could not extract preview'}), 500


@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a processing task"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404

    if processing_status[task_id]['status'] in ['completed', 'error']:
        return jsonify({'error': 'Task already finished'}), 400

    # Mark as cancelled
    processing_status[task_id] = {
        'status': 'cancelled',
        'progress': 0,
        'message': 'Task cancelled by user'
    }

    return jsonify({'message': 'Task cancelled successfully'})


@app.route('/api/cleanup', methods=['POST'])
def cleanup_old_files():
    """Clean up old processed files"""
    try:
        # Delete files older than 1 hour
        current_time = time.time()
        deleted_files = 0

        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    file_age = current_time - os.path.getmtime(file_path)

                    # Delete files older than 1 hour
                    if file_age > 3600:  # 3600 seconds = 1 hour
                        os.remove(file_path)
                        deleted_files += 1
                except:
                    pass

        # Also clean up old status entries
        current_time = time.time()
        old_tasks = []
        for task_id, status in list(processing_status.items()):
            created_at = status.get('created_at', 0)
            if current_time - created_at > 3600:  # Older than 1 hour
                old_tasks.append(task_id)

        for task_id in old_tasks:
            if task_id in processing_status:
                del processing_status[task_id]

        return jsonify({
            'message': f'Cleaned up {deleted_files} old files and {len(old_tasks)} old tasks',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Please download the model first:")
        print("python -c \"from insightface.model_zoo import get_model; get_model('inswapper_128.onnx')\"")

    # Start Flask app
    print("=" * 50)
    print("Starting Face Swapper API (OpenCV Version)")
    print("=" * 50)
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Processed folder: {os.path.abspath(PROCESSED_FOLDER)}")
    print(f"Model path: {model_path}")
    print(f"Max file size: {MAX_CONTENT_LENGTH / (1024 * 1024):.0f} MB")
    print("=" * 50)
    print("\nAvailable endpoints:")
    print("  POST /api/swap-faces    - Start face swapping")
    print("  GET  /api/status/<id>   - Check processing status")
    print("  GET  /api/download/<id> - Download processed video")
    print("  GET  /api/health        - Health check")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)