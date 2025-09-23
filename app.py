import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import json
import time
from datetime import datetime, timedelta
import threading
import queue
from pathlib import Path
import hashlib
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox, Canvas, Label, Frame, Scrollbar, filedialog, Toplevel, Text
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pickle
import requests
import paho.mqtt.client as mqtt

import subprocess
import sys

import logging
from logging.handlers import RotatingFileHandler

# === LOGGER SETUP ===
logger = logging.getLogger('Real-time-Attendance')
default_log_level = logging.INFO
logger.setLevel(default_log_level)

handler = RotatingFileHandler('app.log', maxBytes=1048576, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class AttendanceConfig:
    """Configuration for attendance system"""
    def __init__(self):
        self.MODEL_NAME = "buffalo_l"  # Model for face recognition
        self.MODEL_PATH = ""  # Custom model path if any
        self.EXEC_PROVIDER = ["CPUExecutionProvider"]  # ONNX Runtime providers
        self.ALLOWED_MODULES = ['detection', 'recognition']
        self.MODE = "per_day"  # per_day, per_hour, everytime, 100
        self.CUSTOM_MODE_INTERVAL_IN_SEC = 5  # Used if MODE is custom
        self.SIMILARITY_THRESHOLD = 0.4
        self.UNKNOWN_DIR = "unknown_faces"
        self.EMPLOYEES_DIR = "employees"
        self.ATTENDANCE_LOG = "attendance_log.json"
        self.EMBEDDINGS_CACHE = "embeddings_cache.pkl"
        self.CAMERA_INDEX = 0  # Default to first webcam
        self.DETECTION_CONFIDENCE = 0.5
        self.MAX_FACE_SIZE = 512
        self.UNKNOWN_FACE_LOGGING = False
        self.MAX_UNKNOWN_FACES = 100
        self.UNKNOWN_FULL_IMAGE = True
        self.MAX_UNKNOWN_SAVE_INTERVAL = 10  # seconds
        
        # API/MQTT Settings
        self.API_ENDPOINT = ""
        self.API_KEY = ""
        self.MQTT_BROKER = ""
        self.MQTT_PORT = 1883
        self.MQTT_TOPIC = "attendance/clockin"
        self.MQTT_USERNAME = ""
        self.MQTT_PASSWORD = ""
        self.ENABLE_API = False
        self.ENABLE_MQTT = False

        # Logging config
        self.ENABLE_LOGGING = True
        self.LOG_LEVEL = "INFO"  # Could be DEBUG, INFO, WARNING, ERROR, CRITICAL
        
        self.CONFIG_FILE = "config.json"
        self.load_config()
        self.configure_logging()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        config_data = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                config_data[attr] = getattr(self, attr)
        
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=2)
            logger.info("Configuration saved")

    def configure_logging(self):
        """Enable/disable logging and set log level from config"""
        if not self.ENABLE_LOGGING:
            logger.disabled = True
        else:
            logger.disabled = False
            # Set log level based on config
            log_level_str = self.LOG_LEVEL.upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            logger.setLevel(log_level)
            for h in logger.handlers:
                h.setLevel(log_level)
            logger.info(f"Logging enabled at {log_level_str} level")

class NotificationManager:
    """Manages API and MQTT notifications"""
    def __init__(self, config):
        self.config = config
        self.mqtt_client = None
        self.setup_mqtt()
    
    def setup_mqtt(self):
        """Setup MQTT client"""
        if self.config.ENABLE_MQTT and self.config.MQTT_BROKER:
            try:
                self.mqtt_client = mqtt.Client()
                if self.config.MQTT_USERNAME:
                    self.mqtt_client.username_pw_set(
                        self.config.MQTT_USERNAME, 
                        self.config.MQTT_PASSWORD
                    )
                self.mqtt_client.connect(self.config.MQTT_BROKER, self.config.MQTT_PORT, 60)
                self.mqtt_client.loop_start()
                logger.info("MQTT client connected")
            except Exception as e:
                logger.error(f"MQTT connection error: {e}")
    
    def send_attendance_notification(self, employee_data):
        """Send attendance notification via API and/or MQTT"""
        notification_data = {
            'employee_id': employee_data['employee_id'],
            'timestamp': datetime.now().isoformat(),
            'similarity': float(employee_data.get('similarity', 0))
        }
        # Send API notification
        if self.config.ENABLE_API and self.config.API_ENDPOINT:
            threading.Thread(
                target=self._send_api_notification, 
                args=(notification_data,), 
                daemon=True
            ).start()
        
        # Send MQTT notification
        if self.config.ENABLE_MQTT and self.mqtt_client:
            threading.Thread(
                target=self._send_mqtt_notification, 
                args=(notification_data,), 
                daemon=True
            ).start()
    
    def _send_api_notification(self, data):
        """Send API notification"""
        try:
            headers = {'Content-Type': 'application/json'}
            if self.config.API_KEY:
                headers['Authorization'] = f'Bearer {self.config.API_KEY}'
            
            response = requests.post(
                self.config.API_ENDPOINT, 
                json=data, 
                headers=headers,
                timeout=5
            )
            logger.info(f"API notification sent: {response.status_code}")
        except Exception as e:
            logger.error(f"API notification error: {e}")
    
    def _send_mqtt_notification(self, data):
        """Send MQTT notification"""
        try:
            self.mqtt_client.publish(
                self.config.MQTT_TOPIC, 
                json.dumps(data)
            )
            logger.info("MQTT notification sent")
        except Exception as e:
            logger.error(f"MQTT notification error: {e}")

class EmployeeManager:
    """Manages employee data and embeddings"""
    def __init__(self, config, face_app):
        self.config = config
        self.face_app = face_app
        self.employees = {}
        self.embeddings = {}
        self.folder_hash = None
        self.lock = threading.Lock()
        self.load_employees()
        
    def calculate_folder_hash(self):
        """Calculate hash of employees folder structure"""
        hash_md5 = hashlib.md5()
        if os.path.exists(self.config.EMPLOYEES_DIR):
            for root, dirs, files in os.walk(self.config.EMPLOYEES_DIR):
                for file in sorted(files):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(root, file)
                        if os.path.exists(filepath):
                            hash_md5.update(str(os.path.getmtime(filepath)).encode())
        return hash_md5.hexdigest()
    
    def load_employees(self):
        """Load employee embeddings from folder structure"""
        with self.lock:
            current_hash = self.calculate_folder_hash()
            
            # Check if we can use cached embeddings
            if os.path.exists(self.config.EMBEDDINGS_CACHE):
                try:
                    with open(self.config.EMBEDDINGS_CACHE, 'rb') as f:
                        cache_data = pickle.load(f)
                        if cache_data.get('hash') == current_hash:
                            self.employees = cache_data['employees']
                            self.embeddings = cache_data['embeddings']
                            self.folder_hash = current_hash
                            print(f"Loaded {len(self.employees)} employees from cache")
                            return
                except Exception as e:
                    print(f"Error loading cache: {e}")
            
            # Load fresh embeddings
            self.employees = {}
            self.embeddings = {}
            
            if not os.path.exists(self.config.EMPLOYEES_DIR):
                os.makedirs(self.config.EMPLOYEES_DIR)
                return
            
            for emp_id in os.listdir(self.config.EMPLOYEES_DIR):
                emp_path = os.path.join(self.config.EMPLOYEES_DIR, emp_id)
                if os.path.isdir(emp_path):
                    self.employees[emp_id] = {
                        'id': emp_id,
                        'images': [],
                        'embeddings': []
                    }
                    
                    for img_file in os.listdir(emp_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(emp_path, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                faces = self.face_app.get(img)
                                if faces:
                                    embedding = faces[0].embedding
                                    self.employees[emp_id]['images'].append(img_path)
                                    self.employees[emp_id]['embeddings'].append(embedding)
                                    
                                    if emp_id not in self.embeddings:
                                        self.embeddings[emp_id] = []
                                    self.embeddings[emp_id].append(embedding)
            
            # Save to cache
            self.folder_hash = current_hash
            cache_data = {
                'hash': self.folder_hash,
                'employees': self.employees,
                'embeddings': self.embeddings
            }
            with open(self.config.EMBEDDINGS_CACHE, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Loaded {len(self.employees)} employees with embeddings")
    
    def check_for_updates(self):
        """Check if employees folder has changed"""
        current_hash = self.calculate_folder_hash()
        if current_hash != self.folder_hash:
            print("Employee folder changed, reloading...")
            self.load_employees()
            return True
        return False
    
    def force_reload(self):
        """Force reload of employee embeddings"""
        if os.path.exists(self.config.EMBEDDINGS_CACHE):
            os.remove(self.config.EMBEDDINGS_CACHE)
        self.load_employees()
    
    def add_employee(self, emp_id, image_paths):
        """Add new employee with images"""
        emp_dir = os.path.join(self.config.EMPLOYEES_DIR, emp_id)
        os.makedirs(emp_dir, exist_ok=True)
        
        for i, img_path in enumerate(image_paths):
            if os.path.exists(img_path):
                ext = os.path.splitext(img_path)[1]
                dest_path = os.path.join(emp_dir, f"photo_{i+1}{ext}")
                # Copy image
                import shutil
                shutil.copy2(img_path, dest_path)
        
        self.force_reload()
    
    def add_employee_from_frame(self, emp_id, frame):
        """Add new employee from camera frame"""
        emp_dir = os.path.join(self.config.EMPLOYEES_DIR, emp_id)
        os.makedirs(emp_dir, exist_ok=True)
        
        # Detect face in frame
        faces = self.face_app.get(frame)
        if faces:
            # Get the largest face
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            bbox = largest_face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Extract face with some padding
            padding = 20
            face_img = frame[max(0, y1-padding):min(frame.shape[0], y2+padding), 
                            max(0, x1-padding):min(frame.shape[1], x2+padding)]
            
            # Save face image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(emp_dir, f"photo_{timestamp}.jpg")
            cv2.imwrite(img_path, face_img)
            
            self.force_reload()
            return True
        return False
    
    def remove_employee(self, emp_id):
        """Remove employee and their data"""
        emp_dir = os.path.join(self.config.EMPLOYEES_DIR, emp_id)
        if os.path.exists(emp_dir):
            import shutil
            shutil.rmtree(emp_dir)
            self.force_reload()
    
    def match_face(self, face_embedding):
        """Match face embedding with employee database"""
        best_match = None
        best_similarity = -1
        best_ref_image = None
        
        with self.lock:
            for emp_id, embeddings in self.embeddings.items():
                for idx, ref_embedding in enumerate(embeddings):
                    similarity = np.dot(face_embedding, ref_embedding) / (
                        np.linalg.norm(face_embedding) * np.linalg.norm(ref_embedding)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = emp_id
                        best_ref_image = self.employees[emp_id]['images'][idx]
        
        if best_similarity > self.config.SIMILARITY_THRESHOLD:
            return {
                'employee_id': best_match,
                'similarity': best_similarity,
                'ref_image': best_ref_image
            }
        return None

class AttendanceTracker:
    """Tracks attendance based on configured mode"""
    def __init__(self, config):
        self.config = config
        self.attendance_log = defaultdict(dict)
        self.load_log()
    
    def load_log(self):
        """Load attendance log from file"""
        if os.path.exists(self.config.ATTENDANCE_LOG):
            try:
                with open(self.config.ATTENDANCE_LOG, 'r') as f:
                    self.attendance_log = defaultdict(dict, json.load(f))
            except Exception as e:
                logger.error(f"Error loading attendance log: {e}")
    
    def save_log(self):
        """Save attendance log to file"""
        with open(self.config.ATTENDANCE_LOG, 'w') as f:
            json.dump(dict(self.attendance_log), f, indent=2)
            logger.info("Attendance log saved")
    
    def can_clock_in(self, employee_id):
        """Check if employee can clock in based on mode"""
        now = datetime.now()
        
        if self.config.MODE == "everytime":
            return True
        
        if employee_id not in self.attendance_log:
            return True
        
        last_clock_in = self.attendance_log[employee_id].get('last_clock_in')
        if not last_clock_in:
            return True
        
        last_time = datetime.fromisoformat(last_clock_in)
        
        if self.config.MODE == "per_day":
            return last_time.date() != now.date()
        elif self.config.MODE == "per_hour":
            return (now - last_time).total_seconds() >= 3600
        elif self.config.MODE == "custom":
            return (now - last_time).total_seconds() >= int(self.config.CUSTOM_MODE_INTERVAL_IN_SEC)
        
        return False
    
    def clock_in(self, employee_id):
        """Record attendance for employee"""
        if self.can_clock_in(employee_id):
            now = datetime.now()
            self.attendance_log[employee_id] = {
                'employee_id': employee_id,
                'last_clock_in': now.isoformat(),
                'total_clockins': self.attendance_log[employee_id].get('total_clockins', 0) + 1
            }
            self.save_log()
            return True
        return False

class SettingsDialog:
    """Settings dialog window"""
    def __init__(self, parent, config, callback=None):
        self.parent = parent
        self.config = config
        self.callback = callback
        
        self.dialog = Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("600x700")
        self.dialog.configure(bg='#1e1e2e')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create settings widgets"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # General settings tab
        general_frame = Frame(notebook, bg='#1e1e2e')
        notebook.add(general_frame, text="General")
        
        self.create_general_settings(general_frame)
        
        # Camera settings tab
        camera_frame = Frame(notebook, bg='#1e1e2e')
        notebook.add(camera_frame, text="Camera")
        
        self.create_camera_settings(camera_frame)
        
        # API/MQTT settings tab
        api_frame = Frame(notebook, bg='#1e1e2e')
        notebook.add(api_frame, text="Notifications")
        
        self.create_api_settings(api_frame)
        
        # Buttons
        button_frame = Frame(self.dialog, bg='#1e1e2e')
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(button_frame, text="Save", command=self.save_settings,
                 bg='#a6e3a1', fg='black', font=('Arial', 10, 'bold')).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.dialog.destroy,
                 bg='#f38ba8', fg='white', font=('Arial', 10)).pack(side=tk.RIGHT, padx=5)
    
    def create_general_settings(self, parent):
        """Create general settings widgets"""
        # Mode selection
        Label(parent, text="Attendance Mode:", bg='#1e1e2e', fg='#cdd6f4', 
              font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        self.mode_var = tk.StringVar(value=self.config.MODE)
        mode_frame = Frame(parent, bg='#1e1e2e')
        mode_frame.pack(fill=tk.X, padx=20, pady=5)
        
        for mode in ['per_day', 'per_hour', 'everytime', 'custom']:
            tk.Radiobutton(mode_frame, text=mode.replace('_', ' ').title(), 
                          variable=self.mode_var, value=mode,
                          bg='#1e1e2e', fg='#cdd6f4', selectcolor='#45475a',
                          font=('Arial', 9)).pack(anchor=tk.W)
        
        # Similarity threshold
        Label(parent, text="Similarity Threshold:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        threshold_frame = Frame(parent, bg='#1e1e2e')
        threshold_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=self.config.SIMILARITY_THRESHOLD)
        threshold_scale = tk.Scale(threshold_frame, from_=0.1, to=0.9, resolution=0.05,
                                  orient=tk.HORIZONTAL, variable=self.threshold_var,
                                  bg='#1e1e2e', fg='#cdd6f4', highlightthickness=0)
        threshold_scale.pack(fill=tk.X)
        
        # Directory settings
        Label(parent, text="Directory Settings:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Employees directory
        emp_frame = Frame(parent, bg='#1e1e2e')
        emp_frame.pack(fill=tk.X, padx=20, pady=2)
        Label(emp_frame, text="Employees Dir:", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 9)).pack(side=tk.LEFT)
        self.emp_dir_var = tk.StringVar(value=self.config.EMPLOYEES_DIR)
        tk.Entry(emp_frame, textvariable=self.emp_dir_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 9)).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Unknown faces directory
        unknown_frame = Frame(parent, bg='#1e1e2e')
        unknown_frame.pack(fill=tk.X, padx=20, pady=2)
        Label(unknown_frame, text="Unknown Dir:", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 9)).pack(side=tk.LEFT)
        self.unknown_dir_var = tk.StringVar(value=self.config.UNKNOWN_DIR)
        tk.Entry(unknown_frame, textvariable=self.unknown_dir_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 9)).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Unknown face logging
        self.unknown_logging_var = tk.BooleanVar(value=self.config.UNKNOWN_FACE_LOGGING)
        tk.Checkbutton(parent, text="Enable Unknown Face Logging", 
                      variable=self.unknown_logging_var,
                      bg='#1e1e2e', fg='#cdd6f4', selectcolor='#45475a',
                      font=('Arial', 9)).pack(anchor=tk.W, padx=20, pady=5)
    
    def create_camera_settings(self, parent):
        """Create camera settings widgets"""
        Label(parent, text="Camera Settings:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Camera index/URL
        cam_frame = Frame(parent, bg='#1e1e2e')
        cam_frame.pack(fill=tk.X, padx=20, pady=5)
        Label(cam_frame, text="Camera (Index/URL):", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 9)).pack(anchor=tk.W)
        self.camera_var = tk.StringVar(value=str(self.config.CAMERA_INDEX))
        tk.Entry(cam_frame, textvariable=self.camera_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 9)).pack(fill=tk.X, pady=2)
        
        # Detection confidence
        Label(parent, text="Detection Confidence:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        conf_frame = Frame(parent, bg='#1e1e2e')
        conf_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.confidence_var = tk.DoubleVar(value=self.config.DETECTION_CONFIDENCE)
        conf_scale = tk.Scale(conf_frame, from_=0.1, to=0.9, resolution=0.05,
                             orient=tk.HORIZONTAL, variable=self.confidence_var,
                             bg='#1e1e2e', fg='#cdd6f4', highlightthickness=0)
        conf_scale.pack(fill=tk.X)
    
    def create_api_settings(self, parent):
        """Create API/MQTT settings widgets"""
        # API Settings
        Label(parent, text="API Settings:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        self.api_enabled_var = tk.BooleanVar(value=self.config.ENABLE_API)
        tk.Checkbutton(parent, text="Enable API Notifications", 
                      variable=self.api_enabled_var,
                      bg='#1e1e2e', fg='#cdd6f4', selectcolor='#45475a',
                      font=('Arial', 9)).pack(anchor=tk.W, padx=20, pady=2)
        
        # API Endpoint
        api_frame = Frame(parent, bg='#1e1e2e')
        api_frame.pack(fill=tk.X, padx=20, pady=2)
        Label(api_frame, text="API Endpoint:", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 9)).pack(anchor=tk.W)
        self.api_endpoint_var = tk.StringVar(value=self.config.API_ENDPOINT)
        tk.Entry(api_frame, textvariable=self.api_endpoint_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 9)).pack(fill=tk.X, pady=2)
        
        # API Key
        key_frame = Frame(parent, bg='#1e1e2e')
        key_frame.pack(fill=tk.X, padx=20, pady=2)
        Label(key_frame, text="API Key:", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 9)).pack(anchor=tk.W)
        self.api_key_var = tk.StringVar(value=self.config.API_KEY)
        tk.Entry(key_frame, textvariable=self.api_key_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 9), show='*').pack(fill=tk.X, pady=2)
        
        # MQTT Settings
        Label(parent, text="MQTT Settings:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(20, 5))
        
        self.mqtt_enabled_var = tk.BooleanVar(value=self.config.ENABLE_MQTT)
        tk.Checkbutton(parent, text="Enable MQTT Notifications", 
                      variable=self.mqtt_enabled_var,
                      bg='#1e1e2e', fg='#cdd6f4', selectcolor='#45475a',
                      font=('Arial', 9)).pack(anchor=tk.W, padx=20, pady=2)
        
        # MQTT Broker
        broker_frame = Frame(parent, bg='#1e1e2e')
        broker_frame.pack(fill=tk.X, padx=20, pady=2)
        Label(broker_frame, text="MQTT Broker:", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 9)).pack(anchor=tk.W)
        self.mqtt_broker_var = tk.StringVar(value=self.config.MQTT_BROKER)
        tk.Entry(broker_frame, textvariable=self.mqtt_broker_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 9)).pack(fill=tk.X, pady=2)
        
        # MQTT Port and Topic
        port_topic_frame = Frame(parent, bg='#1e1e2e')
        port_topic_frame.pack(fill=tk.X, padx=20, pady=2)
        
        Label(port_topic_frame, text="Port:", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 9)).pack(side=tk.LEFT)
        self.mqtt_port_var = tk.IntVar(value=self.config.MQTT_PORT)
        tk.Entry(port_topic_frame, textvariable=self.mqtt_port_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 9), width=10).pack(side=tk.LEFT, padx=(5, 10))
        
        Label(port_topic_frame, text="Topic:", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 9)).pack(side=tk.LEFT)
        self.mqtt_topic_var = tk.StringVar(value=self.config.MQTT_TOPIC)
        tk.Entry(port_topic_frame, textvariable=self.mqtt_topic_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 9)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
    
    def save_settings(self):
        """Save settings and close dialog"""
        # Store old values to check for changes
        old_camera = self.config.CAMERA_INDEX
        old_api = self.config.ENABLE_API
        old_mqtt = self.config.ENABLE_MQTT
        old_broker = self.config.MQTT_BROKER
        old_endpoint = self.config.API_ENDPOINT
        
        # Update config
        self.config.MODE = self.mode_var.get()
        self.config.SIMILARITY_THRESHOLD = self.threshold_var.get()
        self.config.EMPLOYEES_DIR = self.emp_dir_var.get()
        self.config.UNKNOWN_DIR = self.unknown_dir_var.get()
        self.config.UNKNOWN_FACE_LOGGING = self.unknown_logging_var.get()
        
        # Camera settings
        camera_val = self.camera_var.get()
        try:
            self.config.CAMERA_INDEX = int(camera_val)
        except ValueError:
            self.config.CAMERA_INDEX = camera_val
        
        self.config.DETECTION_CONFIDENCE = self.confidence_var.get()
        
        # API/MQTT settings
        self.config.ENABLE_API = self.api_enabled_var.get()
        self.config.API_ENDPOINT = self.api_endpoint_var.get()
        self.config.API_KEY = self.api_key_var.get()
        self.config.ENABLE_MQTT = self.mqtt_enabled_var.get()
        self.config.MQTT_BROKER = self.mqtt_broker_var.get()
        self.config.MQTT_PORT = self.mqtt_port_var.get()
        self.config.MQTT_TOPIC = self.mqtt_topic_var.get()
        
        # Check if restart is required
        restart_needed = (
            old_camera != self.config.CAMERA_INDEX or
            old_api != self.config.ENABLE_API or
            old_mqtt != self.config.ENABLE_MQTT or
            old_broker != self.config.MQTT_BROKER or
            old_endpoint != self.config.API_ENDPOINT
        )
        
        # Save config to file
        self.config.save_config()
        
        if self.callback:
            self.callback()
        
        self.dialog.destroy()
        
        if restart_needed:
            result = messagebox.askyesno(
                "Restart Required",
                "Some settings require application restart to take effect.\n\n"
                "Would you like to restart the application now?\n\n"
                "Changes made:\n"
                + ("• Camera settings\n" if old_camera != self.config.CAMERA_INDEX else "")
                + ("• API settings\n" if old_api != self.config.ENABLE_API or old_endpoint != self.config.API_ENDPOINT else "")
                + ("• MQTT settings\n" if old_mqtt != self.config.ENABLE_MQTT or old_broker != self.config.MQTT_BROKER else "")
            )
            
            if result:
                self.restart_application()
        else:
            messagebox.showinfo("Settings", "Settings saved successfully!")

    def restart_application(self):
        """Restart the application"""
        try:
            # Get the current script path
            script_path = os.path.abspath(sys.argv[0])
            
            # Close current application
            self.parent.quit()
            self.parent.destroy()
            
            # Start new instance
            if getattr(sys, 'frozen', False):
                # If running as exe
                subprocess.Popen([sys.executable] + sys.argv[1:])
            else:
                # If running as script
                subprocess.Popen([sys.executable, script_path] + sys.argv[1:])
            
            sys.exit(0)
            
        except Exception as e:
            messagebox.showerror(
                "Restart Failed", 
                f"Failed to restart application: {str(e)}\n\n"
                "Please restart manually to apply all changes."
            )
            logger.error(f"Restart error: {e}")


class UnknownFacesDialog:
    """Dialog to view unknown faces"""
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        
        self.dialog = Toplevel(parent)
        self.dialog.title("Unknown Faces")
        self.dialog.geometry("800x600")
        self.dialog.configure(bg='#1e1e2e')
        self.dialog.transient(parent)
        
        self.create_widgets()
        self.load_unknown_faces()
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Title
        Label(self.dialog, text="Unknown Faces", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Controls
        controls = Frame(self.dialog, bg='#1e1e2e')
        controls.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(controls, text="Refresh", command=self.load_unknown_faces,
                 bg='#89b4fa', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Clear All", command=self.clear_all_unknown,
                 bg='#f38ba8', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        # Scrollable frame
        canvas_frame = Frame(self.dialog, bg='#1e1e2e')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = Canvas(canvas_frame, bg='#181825', highlightthickness=0)
        scrollbar = Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas, bg='#181825')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def load_unknown_faces(self):
        """Load and display unknown faces"""
        try:
            # Clear existing widgets
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            if not os.path.exists(self.config.UNKNOWN_DIR):
                Label(self.scrollable_frame, text="No unknown faces directory found",
                    bg='#181825', fg='#a6adc8', font=('Arial', 12)).pack(pady=50)
                return
            
            unknown_files = [f for f in os.listdir(self.config.UNKNOWN_DIR) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not unknown_files:
                Label(self.scrollable_frame, text="No unknown faces found",
                    bg='#181825', fg='#a6adc8', font=('Arial', 12)).pack(pady=50)
                return
            
            # Display unknown faces in grid
            row = 0
            col = 0
            max_cols = 4
            
            for img_file in sorted(unknown_files, reverse=True):  # Most recent first
                img_path = os.path.join(self.config.UNKNOWN_DIR, img_file)
                self.create_unknown_face_widget(img_path, row, col)
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
        except Exception as e:
            logger.error(f"Error loading unknown faces: {e}")
    
    def create_unknown_face_widget(self, img_path, row, col):
        """Create widget for unknown face"""
        frame = Frame(self.scrollable_frame, bg='#45475a', relief=tk.RAISED, borderwidth=1)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        
        # Load and display image
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((150, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_pil)
                
                img_label = Label(frame, image=photo, bg='#45475a')
                img_label.image = photo
                img_label.pack(pady=5)
        except Exception as e:
            Label(frame, text="Error loading image", bg='#45475a', fg='#f38ba8',
                  font=('Arial', 8)).pack(pady=20)
            logger.error(f"Error displaying image {img_path}: {e}")
        
        # File info
        filename = os.path.basename(img_path)
        Label(frame, text=filename[:20] + ("..." if len(filename) > 20 else ""),
              bg='#45475a', fg='#cdd6f4', font=('Arial', 8)).pack()
        
        # Buttons
        btn_frame = Frame(frame, bg='#45475a')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(btn_frame, text="Add as Employee", 
                 command=lambda: self.add_as_employee(img_path),
                 bg='#a6e3a1', fg='black', font=('Arial', 7)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Delete", 
                 command=lambda: self.delete_unknown(img_path),
                 bg='#f38ba8', fg='white', font=('Arial', 7)).pack(side=tk.RIGHT, padx=2)
    
    def add_as_employee(self, img_path):
        """Add unknown face as new employee"""
        # Simple dialog to get employee ID and name
        dialog = Toplevel(self.dialog)
        dialog.title("Add Employee")
        dialog.geometry("300x150")
        dialog.configure(bg='#1e1e2e')
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        Label(dialog, text="Employee ID:", bg='#1e1e2e', fg='#cdd6f4').pack(pady=5)
        emp_id_var = tk.StringVar()
        tk.Entry(dialog, textvariable=emp_id_var, bg='#45475a', fg='#cdd6f4').pack(pady=5)
        
        def save_employee():
            try:
                emp_id = emp_id_var.get().strip()
                if emp_id:
                    emp_dir = os.path.join(self.config.EMPLOYEES_DIR, emp_id)
                    os.makedirs(emp_dir, exist_ok=True)
                    
                    # Copy image to employee directory
                    import shutil
                    dest_path = os.path.join(emp_dir, f"photo_1{os.path.splitext(img_path)[1]}")
                    shutil.copy2(img_path, dest_path)
                    
                    # Delete from unknown
                    os.remove(img_path)
                    
                    messagebox.showinfo("Success", f"Employee {emp_id} added successfully!")
                    logger.info(f"Added new employee {emp_id}")
                    dialog.destroy()
                    self.load_unknown_faces()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add employee: {e}")
                logger.error(f"Error adding employee: {e}")
            
        tk.Button(dialog, text="Save", command=save_employee,
                 bg='#a6e3a1', fg='black').pack(pady=10)
    
    def delete_unknown(self, img_path):
        """Delete unknown face"""
        if messagebox.askyesno("Confirm", "Delete this unknown face?"):
            os.remove(img_path)
            self.load_unknown_faces()
    
    def clear_all_unknown(self):
        """Clear all unknown faces"""
        if messagebox.askyesno("Confirm", "Delete all unknown faces?"):
            import shutil
            if os.path.exists(self.config.UNKNOWN_DIR):
                shutil.rmtree(self.config.UNKNOWN_DIR)
                os.makedirs(self.config.UNKNOWN_DIR)
            self.load_unknown_faces()

class CameraCaptureDialog:
    """Dialog for capturing employee photo from camera"""
    def __init__(self, parent, face_app, camera_index=0, callback=None):
        self.parent = parent
        self.face_app = face_app
        self.camera_index = camera_index
        self.callback = callback
        self.captured_frame = None
        
        self.dialog = Toplevel(parent)
        self.dialog.title("Capture Employee Photo")
        self.dialog.geometry("800x600")
        self.dialog.configure(bg='#1e1e2e')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        
        self.create_widgets()
        self.start_camera()
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Title
        Label(self.dialog, text="Position your face in the camera and click Capture", 
              bg='#1e1e2e', fg='#cdd6f4', font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Camera frame
        self.camera_canvas = Canvas(self.dialog, bg='#181825', width=640, height=480)
        self.camera_canvas.pack(pady=10)
        
        # Buttons
        btn_frame = Frame(self.dialog, bg='#1e1e2e')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Capture", command=self.capture_photo,
                 bg='#a6e3a1', fg='black', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.close_dialog,
                 bg='#f38ba8', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
        
        # Status
        self.status_label = Label(self.dialog, text="Position your face in the camera", 
                                 bg='#1e1e2e', fg='#a6adc8', font=('Arial', 10))
        self.status_label.pack(pady=5)
    
    def start_camera(self):
        """Start camera feed"""
        self.update_camera()
    
    def update_camera(self):
        """Update camera feed"""
        try:
            if not self.running:
                return
                
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.face_app.get(frame)
                
                # Draw face rectangles
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update status
                if faces:
                    self.status_label.config(text=f"Face detected! Ready to capture.")
                else:
                    self.status_label.config(text="No face detected. Position your face in the camera.")
                
                # Convert to PIL and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                self.camera_canvas.delete("all")
                self.camera_canvas.create_image(320, 240, image=photo, anchor=tk.CENTER)
                self.camera_canvas.image = photo
                
                # Store current frame
                self.current_frame = frame
            
            if self.running:
                self.dialog.after(30, self.update_camera)
        except Exception as e:
            logger.error(f"Camera error: {e}")
            self.status_label.config(text="Error accessing camera.")
            self.running = False
            if self.cap:
                self.cap.release()
    
    def capture_photo(self):
        """Capture current frame"""
        if hasattr(self, 'current_frame'):
            faces = self.face_app.get(self.current_frame)
            if faces:
                self.captured_frame = self.current_frame.copy()
                if self.callback:
                    self.callback(self.captured_frame)
                self.close_dialog()
            else:
                messagebox.showerror("Error", "No face detected! Please position your face in the camera.")
        else:
            messagebox.showerror("Error", "No camera frame available!")
    
    def close_dialog(self):
        """Close dialog and release camera"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.dialog.destroy()

class EmployeeManagementDialog:
    """Dialog for managing employees"""
    def __init__(self, parent, config, employee_manager):
        self.parent = parent
        self.config = config
        self.employee_manager = employee_manager
        
        self.dialog = Toplevel(parent)
        self.dialog.title("Employee Management")
        self.dialog.geometry("800x600")
        self.dialog.configure(bg='#1e1e2e')
        self.dialog.transient(parent)
        
        self.create_widgets()
        self.refresh_employees()
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Title
        Label(self.dialog, text="Employee Management", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Controls
        controls = Frame(self.dialog, bg='#1e1e2e')
        controls.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(controls, text="Add Employee", command=self.add_employee_dialog,
                 bg='#a6e3a1', fg='black', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Refresh", command=self.refresh_employees,
                 bg='#89b4fa', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Recreate Embeddings", command=self.recreate_embeddings,
                 bg='#fab387', fg='black', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        # Employee list
        list_frame = Frame(self.dialog, bg='#1e1e2e')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for employee list
        columns = ('ID', 'Images', 'Actions')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click
        self.tree.bind('<Double-1>', self.on_employee_double_click)
        
        # Context menu
        self.context_menu = tk.Menu(self.dialog, tearoff=0)
        self.context_menu.add_command(label="View Details", command=self.view_employee_details)
        self.context_menu.add_command(label="Add Images", command=self.add_employee_images)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Delete Employee", command=self.delete_employee)
        
        self.tree.bind('<Button-3>', self.show_context_menu)
    
    def refresh_employees(self):
        """Refresh employee list"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add employees
        for emp_id, emp_data in self.employee_manager.employees.items():
            self.tree.insert('', 'end', values=(
                emp_id,
                len(emp_data['images']),
                'Double-click for options'
            ))
    
    def add_employee_dialog(self):
        """Show add employee dialog"""
        print("Add Employee Dialog 1")
        dialog = Toplevel(self.dialog)
        dialog.title("Add New Employee")
        dialog.geometry("500x600")
        dialog.configure(bg='#1e1e2e')
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        # Employee ID
        Label(dialog, text="Employee ID:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        emp_id_var = tk.StringVar()
        tk.Entry(dialog, textvariable=emp_id_var, bg='#45475a', fg='#cdd6f4',
                font=('Arial', 12), width=30).pack(pady=5)
        
        # Method selection
        Label(dialog, text="Choose Method:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        
        method_frame = Frame(dialog, bg='#1e1e2e')
        method_frame.pack(pady=10)
        
        # Capture from camera button
        def capture_from_camera():
            emp_id = emp_id_var.get().strip()
            if not emp_id:
                messagebox.showerror("Error", "Please enter Employee ID first!")
                return
            
            def on_capture(frame):
                try:
                    success = self.employee_manager.add_employee_from_frame(emp_id, frame)
                    if success:
                        messagebox.showinfo("Success", f"Employee {emp_id} added successfully!")
                        dialog.destroy()
                        self.refresh_employees()
                    else:
                        messagebox.showerror("Error", "No face detected in captured image!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to add employee: {str(e)}")
                    logger.error(f"Error adding employee from camera: {e}")
            
            CameraCaptureDialog(self.dialog, self.employee_manager.face_app, self.config.CAMERA_INDEX, on_capture)
        
        tk.Button(method_frame, text="📷 Capture from Camera", 
                 command=capture_from_camera,
                 bg='#89b4fa', fg='white', font=('Arial', 12, 'bold'),
                 width=25, height=2).pack(pady=10)
        
        # OR separator
        Label(dialog, text="- OR -", bg='#1e1e2e', fg='#a6adc8',
              font=('Arial', 10)).pack(pady=10)
        
        # Browse images section
        Label(dialog, text="Select Images from Files:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 12, 'bold')).pack(pady=(10, 5))
        
        images_frame = Frame(dialog, bg='#1e1e2e')
        images_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        images_listbox = tk.Listbox(images_frame, bg='#45475a', fg='#cdd6f4', height=8)
        images_listbox.pack(fill=tk.BOTH, expand=True)
        
        selected_images = []
        
        def browse_images():
            files = filedialog.askopenfilenames(
                title="Select Employee Images",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            for file in files:
                if file not in selected_images:
                    selected_images.append(file)
                    images_listbox.insert(tk.END, os.path.basename(file))
        
        def remove_selected_image():
            selection = images_listbox.curselection()
            if selection:
                index = selection[0]
                images_listbox.delete(index)
                selected_images.pop(index)
        
        browse_frame = Frame(dialog, bg='#1e1e2e')
        browse_frame.pack(pady=5)
        
        tk.Button(browse_frame, text="Browse Images", command=browse_images,
                 bg='#fab387', fg='black', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(browse_frame, text="Remove Selected", command=remove_selected_image,
                 bg='#f38ba8', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        def save_employee_with_files():
            emp_id = emp_id_var.get().strip()
            
            if not emp_id:
                messagebox.showerror("Error", "Employee ID is required!")
                return
            
            if not selected_images:
                messagebox.showerror("Error", "Please select at least one image!")
                return
            
            # Check if employee already exists
            if emp_id in self.employee_manager.employees:
                messagebox.showerror("Error", "Employee ID already exists!")
                return
            
            try:
                self.employee_manager.add_employee(emp_id, selected_images)
                messagebox.showinfo("Success", f"Employee {emp_id} added successfully!")
                logger.info(f"Added new employee {emp_id} with {len(selected_images)} images")
                dialog.destroy()
                self.refresh_employees()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add employee: {str(e)}")
                logger.error(f"Error adding employee from files: {e}")
        
        # Buttons
        btn_frame = Frame(dialog, bg='#1e1e2e')
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Button(btn_frame, text="Save with Selected Files", command=save_employee_with_files,
                 bg='#a6e3a1', fg='black', font=('Arial', 12, 'bold')).pack(side=tk.RIGHT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy,
                 bg='#f38ba8', fg='white', font=('Arial', 12)).pack(side=tk.RIGHT, padx=5)
        
        # dialog = Toplevel(self.dialog)
        # dialog.title("Add New Employee")
        # dialog.geometry("400x300")
        # dialog.configure(bg='#1e1e2e')
        # dialog.transient(self.dialog)
        # dialog.grab_set()
        
        # # Employee ID
        # Label(dialog, text="Employee ID:", bg='#1e1e2e', fg='#cdd6f4',
        #       font=('Arial', 10, 'bold')).pack(pady=5)
        # emp_id_var = tk.StringVar()
        # tk.Entry(dialog, textvariable=emp_id_var, bg='#45475a', fg='#cdd6f4',
        #         font=('Arial', 10)).pack(pady=5, padx=20, fill=tk.X)
        
                        
    def on_employee_double_click(self, event):
        """Handle employee double-click"""
        item = self.tree.selection()[0]
        emp_id = self.tree.item(item, 'values')[0]
        self.view_employee_details_for_id(emp_id)
    
    def show_context_menu(self, event):
        """Show context menu"""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
    
    def view_employee_details(self):
        """View employee details"""
        selection = self.tree.selection()
        if selection:
            emp_id = self.tree.item(selection[0], 'values')[0]
            self.view_employee_details_for_id(emp_id)
    
    def view_employee_details_for_id(self, emp_id):
        """View details for specific employee"""
        if emp_id not in self.employee_manager.employees:
            return
        
        emp_data = self.employee_manager.employees[emp_id]
        
        dialog = Toplevel(self.dialog)
        dialog.title(f"Employee Details - {emp_id}")
        dialog.geometry("800x600")
        dialog.configure(bg='#1e1e2e')
        dialog.transient(self.dialog)
        
        # Info frame
        info_frame = Frame(dialog, bg='#1e1e2e')
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        Label(info_frame, text=f"Employee ID: {emp_data['id']}", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 14, 'bold')).pack(anchor=tk.W)
        Label(info_frame, text=f"Total Images: {len(emp_data['images'])}", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 12)).pack(anchor=tk.W)
        
        # Action buttons
        action_frame = Frame(dialog, bg='#1e1e2e')
        action_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def add_image_from_camera():
            def on_capture(frame):
                try:
                    # Detect face in frame
                    faces = self.employee_manager.face_app.get(frame)
                    if faces:
                        # Get the largest face
                        largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        bbox = largest_face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        
                        # Extract face with padding
                        padding = 20
                        face_img = frame[max(0, y1-padding):min(frame.shape[0], y2+padding), 
                                        max(0, x1-padding):min(frame.shape[1], x2+padding)]
                        
                        # Save face image
                        emp_dir = os.path.join(self.config.EMPLOYEES_DIR, emp_id)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(emp_dir, f"photo_{timestamp}.jpg")
                        cv2.imwrite(img_path, face_img)
                        
                        self.employee_manager.force_reload()
                        messagebox.showinfo("Success", "Image added successfully!")
                        dialog.destroy()
                        self.view_employee_details_for_id(emp_id)  # Refresh dialog
                    else:
                        messagebox.showerror("Error", "No face detected in captured image!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to add image: {str(e)}")
                    logger.error(f"Error adding image from camera: {e}")
            
            CameraCaptureDialog(self.dialog, self.employee_manager.face_app, self.config.CAMERA_INDEX, on_capture)
        
        def add_images_from_files():
            files = filedialog.askopenfilenames(
                title=f"Add images for {emp_id}",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            
            if files:
                emp_dir = os.path.join(self.config.EMPLOYEES_DIR, emp_id)
                
                for file in files:
                    # Copy image to employee directory
                    import shutil
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"photo_{timestamp}{os.path.splitext(file)[1]}"
                    dest_path = os.path.join(emp_dir, filename)
                    shutil.copy2(file, dest_path)
                
                # Recreate embeddings
                self.employee_manager.force_reload()
                messagebox.showinfo("Success", f"Added {len(files)} images!")
                dialog.destroy()
                self.view_employee_details_for_id(emp_id)  # Refresh dialog
        
        tk.Button(action_frame, text="📷 Add Image from Camera", command=add_image_from_camera,
                 bg='#89b4fa', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="📁 Add Images from Files", command=add_images_from_files,
                 bg='#fab387', fg='black', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Images section
        Label(dialog, text="Employee Images:", bg='#1e1e2e', fg='#cdd6f4',
              font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        
        # Scrollable frame for images
        images_main_frame = Frame(dialog, bg='#1e1e2e')
        images_main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        canvas = Canvas(images_main_frame, bg='#181825', highlightthickness=0)
        scrollbar_v = Scrollbar(images_main_frame, orient="vertical", command=canvas.yview)
        scrollbar_h = Scrollbar(images_main_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = Frame(canvas, bg='#181825')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # Display images in grid
        max_cols = 4
        if emp_data['images']:
            row = 0
            col = 0
            
            for img_path in emp_data['images']:
                if os.path.exists(img_path):
                    try:
                        # Create frame for each image
                        img_frame = Frame(scrollable_frame, bg='#45475a', relief=tk.RAISED, borderwidth=2)
                        img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                        
                        # Load and display image
                        img = cv2.imread(img_path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(img_rgb)
                            img_pil.thumbnail((150, 150), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img_pil)
                            
                            img_label = Label(img_frame, image=photo, bg='#45475a')
                            img_label.image = photo  # Keep reference
                            img_label.pack(pady=5)
                            
                            # Image info
                            filename = os.path.basename(img_path)
                            Label(img_frame, text=filename[:15] + ("..." if len(filename) > 15 else ""),
                                  bg='#45475a', fg='#cdd6f4', font=('Arial', 8)).pack()
                            
                            # Delete button
                            def delete_image(path=img_path):
                                if messagebox.askyesno("Confirm", f"Delete {os.path.basename(path)}?"):
                                    try:
                                        os.remove(path)
                                        self.employee_manager.force_reload()
                                        messagebox.showinfo("Success", "Image deleted!")
                                        dialog.destroy()
                                        self.view_employee_details_for_id(emp_id)  # Refresh dialog
                                    except Exception as e:
                                        messagebox.showerror("Error", f"Failed to delete image: {str(e)}")
                                        logger.error(f"Error deleting image: {e} of employee {emp_id}")
                            
                            tk.Button(img_frame, text="🗑️ Delete", command=delete_image,
                                     bg='#f38ba8', fg='white', font=('Arial', 7)).pack(pady=2)
                        
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
                            
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        logger.error(f"Error loading image {img_path}: {e}")
        else:
            Label(scrollable_frame, text="No images found for this employee",
                  bg='#181825', fg='#a6adc8', font=('Arial', 12)).pack(pady=50)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_v.pack(side="right", fill="y")
        scrollbar_h.pack(side="bottom", fill="x")
        
        # Configure grid weights for proper resizing
        for i in range(max_cols):
            scrollable_frame.grid_columnconfigure(i, weight=1)
    
    def add_employee_images(self):
        """Add more images to existing employee"""
        selection = self.tree.selection()
        if not selection:
            return
        
        emp_id = self.tree.item(selection[0], 'values')[0]
        
        files = filedialog.askopenfilenames(
            title=f"Add images for {emp_id}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if files:
            emp_dir = os.path.join(self.config.EMPLOYEES_DIR, emp_id)
            
            for file in files:
                # Copy image to employee directory
                import shutil
                filename = os.path.basename(file)
                dest_path = os.path.join(emp_dir, filename)
                shutil.copy2(file, dest_path)
            
            # Recreate embeddings
            self.employee_manager.force_reload()
            self.refresh_employees()
            messagebox.showinfo("Success", f"Added {len(files)} images for {emp_id}")
            logger.info(f"Added {len(files)} images for employee {emp_id}")
    
    def delete_employee(self):
        """Delete employee"""
        selection = self.tree.selection()
        if not selection:
            return
        
        emp_id = self.tree.item(selection[0], 'values')[0]
        
        if messagebox.askyesno("Confirm Deletion", f"Delete employee {emp_id}?"):
            try:
                self.employee_manager.remove_employee(emp_id)
                self.refresh_employees()
                messagebox.showinfo("Success", f"Employee {emp_id} deleted successfully!")
                logger.info(f"Deleted employee {emp_id}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete employee: {str(e)}")
                logger.error(f"Error deleting employee {emp_id}: {e}")
    
    def recreate_embeddings(self):
        """Recreate all embeddings"""
        if messagebox.askyesno("Confirm", "Recreate all employee embeddings? This may take a while."):
            try:
                self.employee_manager.force_reload()
                self.refresh_employees()
                messagebox.showinfo("Success", "Embeddings recreated successfully!")
                logger.info("Recreated all employee embeddings")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to recreate embeddings: {str(e)}")
                logger.error(f"Error recreating embeddings: {e}")

class AttendanceUI:
    """Beautiful UI for attendance system"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-Time Attendance System")
        self.root.geometry("1400x800")
        self.root.configure(bg='#1e1e2e')
        self.LAST_UNKNOWN_FACE_SAVED= datetime.now()
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', background='#1e1e2e', foreground='#cdd6f4', font=('Arial', 16, 'bold'))
        style.configure('Info.TLabel', background='#1e1e2e', foreground='#a6adc8', font=('Arial', 10))
        
        self.config = AttendanceConfig()
        self.face_app = FaceAnalysis(
            name=self.config.MODEL_NAME, 
            providers=self.config.EXEC_PROVIDER,
            allowed_modules=self.config.ALLOWED_MODULES
            )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.employee_manager = EmployeeManager(self.config, self.face_app)
        self.attendance_tracker = AttendanceTracker(self.config)
        self.notification_manager = NotificationManager(self.config)
        
        self.video_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = True
        
        self.setup_ui()
        self.start_threads()
    
    def setup_ui(self):
        """Setup the UI components"""
        # Menu bar
        self.create_menu_bar()
        
        # Main container
        main_frame = Frame(self.root, bg='#1e1e2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video feed
        left_panel = Frame(main_frame, bg='#313244', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Title for video feed
        video_title = Label(left_panel, text="Live Camera Feed", bg='#313244', fg='#cdd6f4', 
                           font=('Arial', 14, 'bold'))
        video_title.pack(pady=10)
        
        # Video canvas
        self.video_canvas = Canvas(left_panel, bg='#181825', highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Status bar
        self.status_label = Label(left_panel, text="System Ready", bg='#45475a', fg='#a6e3a1',
                                 font=('Arial', 10), relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Right panel - Matched employees
        right_panel = Frame(main_frame, bg='#313244', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Title for matches
        matches_title = Label(right_panel, text="Attendance Records", bg='#313244', fg='#cdd6f4',
                             font=('Arial', 14, 'bold'))
        matches_title.pack(pady=10)
        
        # Controls frame
        controls_frame = Frame(right_panel, bg='#313244')
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Mode selector
        Label(controls_frame, text="Mode:", bg='#313244', fg='#a6adc8').pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value=self.config.MODE)
        mode_menu = ttk.Combobox(controls_frame, textvariable=self.mode_var, 
                                 values=['per_day', 'per_hour', 'everytime', 'custom'], width=10)
        mode_menu.pack(side=tk.LEFT, padx=5)
        mode_menu.bind('<<ComboboxSelected>>', self.change_mode)
        
        # Clear button
        clear_btn = tk.Button(controls_frame, text="Clear History", bg='#f38ba8', fg='white',
                             command=self.clear_history, font=('Arial', 9))
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # Scrollable frame for matches
        canvas_frame = Frame(right_panel, bg='#313244')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.matches_canvas = Canvas(canvas_frame, bg='#181825', highlightthickness=0)
        scrollbar = Scrollbar(canvas_frame, orient="vertical", command=self.matches_canvas.yview)
        self.scrollable_frame = Frame(self.matches_canvas, bg='#181825')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.matches_canvas.configure(scrollregion=self.matches_canvas.bbox("all"))
        )
        
        self.matches_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.matches_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.matches_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.match_widgets = []
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Restart Application", command=self.restart_application)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Employees menu
        employees_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Employees", menu=employees_menu)
        employees_menu.add_command(label="Manage Employees", command=self.open_employee_management)
        employees_menu.add_command(label="View Unknown Faces", command=self.open_unknown_faces)
        employees_menu.add_separator()
        employees_menu.add_command(label="Recreate Embeddings", command=self.recreate_embeddings)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Attendance Log", command=self.view_attendance_log)
        view_menu.add_command(label="System Status", command=self.show_system_status)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def open_settings(self):
        """Open settings dialog"""
        SettingsDialog(self.root, self.config, self.apply_settings)
    
    def apply_settings(self):
        """Apply new settings"""
        # Reinitialize notification manager
        self.notification_manager = NotificationManager(self.config)
        
        # Update mode
        self.mode_var.set(self.config.MODE)
        
        # Update status
        self.status_label.config(text="Settings applied successfully")
    
    def open_employee_management(self):
        """Open employee management dialog"""
        EmployeeManagementDialog(self.root, self.config, self.employee_manager)
    
    def open_unknown_faces(self):
        """Open unknown faces dialog"""
        UnknownFacesDialog(self.root, self.config)
    
    def recreate_embeddings(self):
        """Recreate embeddings"""
        if messagebox.askyesno("Confirm", "Recreate all employee embeddings? This may take a while."):
            self.status_label.config(text="Recreating embeddings...")
            self.root.update()
            
            try:
                self.employee_manager.force_reload()
                self.status_label.config(text="Embeddings recreated successfully")
                messagebox.showinfo("Success", "Embeddings recreated successfully!")
                logger.info("Recreated all employee embeddings")
            except Exception as e:
                self.status_label.config(text="Error recreating embeddings")
                messagebox.showerror("Error", f"Failed to recreate embeddings: {str(e)}")
                logger.error(f"Error recreating embeddings: {e}")
    
    def restart_application(self):
        """Restart the application"""
        result = messagebox.askyesno(
            "Restart Application",
            "Are you sure you want to restart the application?\n\n"
            "This will close the current session and start a new one."
        )
        
        if result:
            try:
                # Get the current script path
                script_path = os.path.abspath(sys.argv[0])
                
                # Close current application
                self.running = False
                time.sleep(0.5)  # Give threads time to stop
                
                # Cleanup MQTT connection
                if hasattr(self.notification_manager, 'mqtt_client') and self.notification_manager.mqtt_client:
                    try:
                        self.notification_manager.mqtt_client.loop_stop()
                        self.notification_manager.mqtt_client.disconnect()
                    except:
                        pass
                
                self.root.quit()
                
                # Start new instance
                if getattr(sys, 'frozen', False):
                    # If running as exe
                    subprocess.Popen([sys.executable] + sys.argv[1:])
                else:
                    # If running as script
                    subprocess.Popen([sys.executable, script_path] + sys.argv[1:])
                
                sys.exit(0)
                
            except Exception as e:
                messagebox.showerror(
                    "Restart Failed", 
                    f"Failed to restart application: {str(e)}\n\n"
                    "Please restart manually to apply all changes."
                )
                logger.error(f"Error restarting application: {e}")
    
    def view_attendance_log(self):
        """View attendance log"""
        dialog = Toplevel(self.root)
        dialog.title("Attendance Log")
        dialog.geometry("800x600")
        dialog.configure(bg='#1e1e2e')
        dialog.transient(self.root)
        
        # Create text widget with scrollbar
        text_frame = Frame(dialog, bg='#1e1e2e')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = Text(text_frame, bg='#181825', fg='#cdd6f4', font=('Courier', 10))
        scrollbar = Scrollbar(text_frame, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Load attendance log
        if os.path.exists(self.config.ATTENDANCE_LOG):
            with open(self.config.ATTENDANCE_LOG, 'r') as f:
                log_data = json.load(f)
                
            for emp_id, data in log_data.items():
                text_widget.insert(tk.END, f"Employee ID: {emp_id}\n")
                text_widget.insert(tk.END, f"Last Clock-in: {data.get('last_clock_in', 'N/A')}\n")
                text_widget.insert(tk.END, f"Total Clock-ins: {data.get('total_clockins', 0)}\n")
                text_widget.insert(tk.END, "-" * 50 + "\n\n")
        else:
            text_widget.insert(tk.END, "No attendance log found.")
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(state=tk.DISABLED)
    
    def show_system_status(self):
        """Show system status"""
        dialog = Toplevel(self.root)
        dialog.title("System Status")
        dialog.geometry("500x400")
        dialog.configure(bg='#1e1e2e')
        dialog.transient(self.root)
        
        # Status information
        status_text = f"""
System Status Report
==================

Configuration:
- Mode: {self.config.MODE}
- Model: {self.config.MODEL_NAME}
- Execution Provider: {', '.join(self.config.EXEC_PROVIDER)}
- Allowed Modules: {', '.join(self.config.ALLOWED_MODULES)}
- Similarity Threshold: {self.config.SIMILARITY_THRESHOLD}
- Camera: {self.config.CAMERA_INDEX}
- Detection Confidence: {self.config.DETECTION_CONFIDENCE}

Employees:
- Total Employees: {len(self.employee_manager.employees)}
- Total Embeddings: {sum(len(emb) for emb in self.employee_manager.embeddings.values())}

Directories:
- Employees Dir: {self.config.EMPLOYEES_DIR}
- Unknown Dir: {self.config.UNKNOWN_DIR}
- Unknown Logging: {'Enabled' if self.config.UNKNOWN_FACE_LOGGING else 'Disabled'}
- Unknown Max Photo: {self.config.MAX_UNKNOWN_FACES}

Notifications:
- API Enabled: {'Yes' if self.config.ENABLE_API else 'No'}
- MQTT Enabled: {'Yes' if self.config.ENABLE_MQTT else 'No'}

System:
- Face App Loaded: {'Yes' if self.face_app else 'No'}
- System Running: {'Yes' if self.running else 'No'}
"""
        
        text_widget = Text(dialog, bg='#181825', fg='#cdd6f4', font=('Courier', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, status_text)
        text_widget.config(state=tk.DISABLED)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Real-Time Attendance System\n"
            "Version 2.0\n\n"
            "Features:\n"
            "- Real-time face recognition\n"
            "- Employee management\n"
            "- Unknown face logging\n"
            "- API/MQTT notifications\n"
            "- Configurable settings\n\n"
            "Built by almamunb72@gmail.com"
        )
    
    def change_mode(self, event=None):
        """Change attendance mode"""
        self.config.MODE = self.mode_var.get()
        self.config.save_config()
        self.status_label.config(text=f"Mode changed to: {self.config.MODE}")
        logger.info(f"Attendance mode changed to: {self.config.MODE}")
    
    def clear_history(self):
        """Clear match history display"""
        for widget in self.match_widgets:
            widget.destroy()
        self.match_widgets.clear()
        logger.info("Cleared attendance history display")
    
    def video_capture_thread(self):
        """Thread for capturing video frames"""
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                try:
                    self.video_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
    
    def face_processing_thread(self):
        """Thread for processing faces"""
        last_update_check = time.time()
        
        while self.running:
            try:
                frame = self.video_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Check for employee folder updates every 5 seconds
            if time.time() - last_update_check > 5:
                self.employee_manager.check_for_updates()
                last_update_check = time.time()
            
            # Detect faces
            faces = self.face_app.get(frame)
            
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                
                # Match face
                match = self.employee_manager.match_face(embedding)
                
                if match:
                    # Try to clock in
                    clocked_in = self.attendance_tracker.clock_in(
                        match['employee_id']
                    )
                    
                    # Send notification if clocked in
                    if clocked_in:
                        self.notification_manager.send_attendance_notification(match)
                    
                    # Extract face image
                    x1, y1, x2, y2 = bbox
                    face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                                    max(0, x1):min(frame.shape[1], x2)]
                    
                    results.append({
                        'bbox': bbox,
                        'match': match,
                        'clocked_in': clocked_in,
                        'face_img': face_img,
                        'timestamp': datetime.now()
                    })
                else:
                    # Save unknown face if logging enabled
                    if self.config.UNKNOWN_FACE_LOGGING and \
                        datetime.now() - self.LAST_UNKNOWN_FACE_SAVED > timedelta(seconds=self.config.MAX_UNKNOWN_SAVE_INTERVAL):
                        try:
                            x1, y1, x2, y2 = bbox
                            
                            if self.config.UNKNOWN_FULL_IMAGE:
                                face_img = frame
                            else:
                                face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                                                max(0, x1):min(frame.shape[1], x2)]
                            
                            if not os.path.exists(self.config.UNKNOWN_DIR):
                                os.makedirs(self.config.UNKNOWN_DIR)

                            # Enforce max photo count
                            existing_photos = sorted(
                                [os.path.join(self.config.UNKNOWN_DIR, f) 
                                for f in os.listdir(self.config.UNKNOWN_DIR) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                                key=os.path.getctime  # sort by creation time
                            )

                            # Delete oldest files if exceeding limit
                            max_photos = self.config.MAX_UNKNOWN_FACES
                            if len(existing_photos) >= max_photos:
                                files_to_delete = existing_photos[:len(existing_photos) - max_photos + 1]
                                for old_file in files_to_delete:
                                    try:
                                        os.remove(old_file)
                                    except Exception as e:
                                        print(f"Failed to delete {old_file}: {e}")
                            
                            self.LAST_UNKNOWN_FACE_SAVED = datetime.now()
                            timestamp = self.LAST_UNKNOWN_FACE_SAVED.strftime("%Y%m%d_%H%M_%f")
                            unknown_path = os.path.join(self.config.UNKNOWN_DIR, f"{timestamp}.jpg")
                            cv2.imwrite(unknown_path, face_img)
                            logger.info(f"Saved unknown face to {unknown_path} at {timestamp}")
                            results.append({
                                'bbox': bbox,
                                'match': None,
                                'face_img': face_img,
                                'timestamp': datetime.now()
                            })
                        except Exception as e:
                            logger.error(f"Error saving unknown face: {e}")
            
            # Send results to UI
            try:
                self.result_queue.put((frame, results), timeout=0.1)
            except queue.Full:
                pass
    
    def update_ui(self):
        """Update UI with processed results"""
        try:
            frame, results = self.result_queue.get_nowait()
            
            # Draw bboxes and labels on frame
            for result in results:
                bbox = result['bbox']
                x1, y1, x2, y2 = bbox
                
                if result['match']:
                    color = (0, 255, 0) if result.get('clocked_in') else (255, 255, 0)
                    label = f"{result['match']['employee_id']} ({result['match']['similarity']:.2f})"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
            
            # Convert frame to PIL Image and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )
            self.video_canvas.image = photo
            
            # Update matches panel
            for result in results:
                if result['match'] and result.get('clocked_in'):
                    self.add_match_widget(result)
            
            # Update status
            if results:
                faces_count = len(results)
                matched_count = sum(1 for r in results if r['match'])
                self.status_label.config(
                    text=f"Detected: {faces_count} faces | Matched: {matched_count}"
                )
        
        except queue.Empty:
            pass
        
        if self.running:
            self.root.after(30, self.update_ui)
    
    def add_match_widget(self, result):
        """Add a match widget to the right panel"""
        # Create match frame
        match_frame = Frame(self.scrollable_frame, bg='#45475a', relief=tk.RAISED, borderwidth=1)
        match_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Info section
        info_frame = Frame(match_frame, bg='#45475a')
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        Label(info_frame, text=result['match']['employee_id'], bg='#45475a', fg='#a6e3a1',
              font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        Label(info_frame, text=f"ID: {result['match']['employee_id']}", bg='#45475a', 
              fg='#cdd6f4', font=('Arial', 9)).pack(anchor=tk.W)
        Label(info_frame, text=f"Similarity: {result['match']['similarity']:.2%}", bg='#45475a',
              fg='#cdd6f4', font=('Arial', 9)).pack(anchor=tk.W)
        Label(info_frame, text=f"Time: {result['timestamp'].strftime('%H:%M:%S')}", bg='#45475a',
              fg='#cdd6f4', font=('Arial', 9)).pack(anchor=tk.W)
        
        # Images section
        images_frame = Frame(match_frame, bg='#45475a')
        images_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Captured face
        if result['face_img'].size > 0:
            face_rgb = cv2.cvtColor(result['face_img'], cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_pil.thumbnail((60, 60), Image.Resampling.LANCZOS)
            face_photo = ImageTk.PhotoImage(face_pil)
            
            face_label = Label(images_frame, image=face_photo, bg='#45475a')
            face_label.image = face_photo
            face_label.pack(side=tk.LEFT, padx=2)
        
        # Reference image
        if result['match']['ref_image'] and os.path.exists(result['match']['ref_image']):
            ref_img = cv2.imread(result['match']['ref_image'])
            if ref_img is not None:
                ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_pil = Image.fromarray(ref_rgb)
                ref_pil.thumbnail((60, 60), Image.Resampling.LANCZOS)
                ref_photo = ImageTk.PhotoImage(ref_pil)
                
                ref_label = Label(images_frame, image=ref_photo, bg='#45475a')
                ref_label.image = ref_photo
                ref_label.pack(side=tk.LEFT, padx=2)
        
        self.match_widgets.append(match_frame)
        
        # Keep only last 10 matches
        if len(self.match_widgets) > 10:
            self.match_widgets[0].destroy()
            self.match_widgets.pop(0)
    
    def start_threads(self):
        """Start background threads"""
        self.video_thread = threading.Thread(target=self.video_capture_thread, daemon=True)
        self.video_thread.start()
        
        self.processing_thread = threading.Thread(target=self.face_processing_thread, daemon=True)
        self.processing_thread.start()
        
        self.update_ui()
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        time.sleep(0.5)  # Give threads time to stop
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create necessary directories
        for directory in [self.config.EMPLOYEES_DIR, self.config.UNKNOWN_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Show instructions
        if not os.listdir(self.config.EMPLOYEES_DIR):
            messagebox.showinfo(
                "Setup Required",
                "Please add employee photos to the 'employees' folder:\n"
                "employees/{employee_id}/{image}.jpg\n\n"
                "Example: employees/john_doe/photo1.jpg\n\n"
                "Or use the Employee Management menu to add employees."
            )
        
        self.root.mainloop()

if __name__ == "__main__":
    app = AttendanceUI()
    app.run()