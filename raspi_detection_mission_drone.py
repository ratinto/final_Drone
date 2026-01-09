#!/usr/bin/env python3
"""
Raspberry Pi Detection Mission Drone
Autonomous mission with human detection and database integration
For real drone with USB MAVLink connection

HEADLESS MODE - No display required
Perfect for SSH operation or background processes
"""

from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import threading
import argparse
import os
from pymavlink import mavutil
from ultralytics import YOLO
import cv2
import requests

# ---------- Configuration ----------
# Connection Settings
DEFAULT_CONNECTION_STRING = "/dev/ttyACM0"  # USB MAVLink connection
DEFAULT_BAUD_RATE = 57600

# YOLO Model
MODEL_PATH = "best.pt"  # Your trained human detection model
CONFIDENCE_THRESHOLD = 0.4

# Camera Settings
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 20

# API Configuration
API_BASE_URL = "https://server-drone.vercel.app/api"

# RC Override Settings
# Channel 5: Flight modes (configured in Pixhawk) - STABILIZE, ALT_HOLD, LOITER, etc.
# Channel 6: Manual/Auto mission toggle (high = manual override, low = auto mission)
RC_CHANNEL_6_THRESHOLD = 1600  # Channel 6 value above which manual mode is active
RC_OVERRIDE_CHECK_INTERVAL = 0.5  # seconds - how often to check for RC override

# Detection Settings
DETECT_CLASS = "human"  # Class name in best.pt model
INVESTIGATION_DISTANCE = 3.0  # meters - how close to approach human
INVESTIGATION_TIME = 5  # seconds - time to hover near human
COOLDOWN_PERIOD = 60  # seconds - don't re-investigate same human
GPS_EXCLUSION_RADIUS = 15  # meters - ignore humans near recent investigations

# Alignment Settings (for centering human during investigation)
ENABLE_ALIGNMENT = True  # Enable centering human in frame during investigation
CENTER_TOLERANCE_X = 40  # pixels - horizontal tolerance
CENTER_TOLERANCE_Y = 40  # pixels - vertical tolerance
MAX_ALIGNMENT_SPEED = 0.5  # m/s - maximum alignment speed
MIN_ALIGNMENT_SPEED = 0.1  # m/s - minimum alignment speed
ALIGNMENT_SPEED_MULTIPLIER = 1.5  # sensitivity of movement
MAX_ALIGNMENT_TIME = 10  # seconds - maximum time spent aligning

# Mission Settings
MISSION_ALTITUDE = 10  # meters
MISSION_SPEED = 3  # m/s
LOG_FILE = "detection_mission_log.txt"

# Headless Mode (No display - perfect for SSH)
HEADLESS = True  # Always runs headless - no GUI windows

# ---------- Global Variables ----------
human_detected = False
human_location = None
detection_lock = threading.Lock()
tracked_ids = set()
total_human_count = 0
mission_paused = False
investigated_humans = {}
investigated_gps_locations = []
log_file = None

# Camera properties (set after initialization)
frame_center_x = 0
frame_center_y = 0
actual_width = 0
actual_height = 0

# ---------- Setup Logging ----------
log_file = open(LOG_FILE, 'a')
log_file.write(f"\n{'='*70}\n")
log_file.write(f"Mission started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"{'='*70}\n")
log_file.flush()

def log_message(message):
    """Log message to console and file"""
    print(message)
    if log_file:
        log_file.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
        log_file.flush()

# ---------- Parse Command Line Arguments ----------
parser = argparse.ArgumentParser(description='Raspberry Pi Detection Mission Drone')
parser.add_argument('--connect', 
                   default=DEFAULT_CONNECTION_STRING,
                   help='Vehicle connection string (default: /dev/ttyACM0)')
parser.add_argument('--baud',
                   type=int,
                   default=DEFAULT_BAUD_RATE,
                   help='Baud rate (default: 57600)')
parser.add_argument('--api',
                   default=API_BASE_URL,
                   help='API base URL')
parser.add_argument('--altitude',
                   type=float,
                   default=MISSION_ALTITUDE,
                   help='Mission altitude in meters (default: 10)')
parser.add_argument('--speed',
                   type=float,
                   default=MISSION_SPEED,
                   help='Mission speed in m/s (default: 3)')

args = parser.parse_args()
API_BASE_URL = args.api
MISSION_ALTITUDE = args.altitude
MISSION_SPEED = args.speed

# ---------- Connect to Vehicle ----------
log_message(f"\n🔌 Connecting to vehicle on: {args.connect}")

try:
    log_message("⏳ Connecting...")
    vehicle = connect(args.connect, baud=args.baud, wait_ready=True, timeout=60)
    log_message("✅ Vehicle connected!")
    log_message(f"   Battery: {vehicle.battery.level}% | Mode: {vehicle.mode.name} | GPS: {vehicle.gps_0.fix_type}")
    
    # Set default speeds
    log_message(f"⚡ Setting default speed to {MISSION_SPEED} m/s...")
    vehicle.groundspeed = MISSION_SPEED
    vehicle.airspeed = MISSION_SPEED
    log_message(f"✅ Speed configured")
    
except Exception as e:
    log_message(f"❌ Connection failed: {e}")
    log_message(f"💡 Check: USB cable, device permissions, baud rate")
    exit(1)

# ---------- Initialize YOLO Model ----------
log_message("🎯 Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    log_message(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    log_message(f"❌ Failed to load model: {e}")
    exit(1)

# ---------- Helper Functions ----------
def check_rc_override():
    """Check if pilot is taking manual control via RC transmitter
    
    Returns:
        True if manual override is active (either via Channel 6 or non-GUIDED/AUTO mode)
        False if autonomous mission should continue
    """
    try:
        # Check RC channel 6 (Manual/Auto toggle)
        # High position (>1600) = Manual override
        # Low position (<1600) = Auto mission
        rc6 = vehicle.channels['6']
        
        if rc6 > RC_CHANNEL_6_THRESHOLD:
            return True  # Manual mode via Channel 6
        
        # Check if flight mode was changed via Channel 5 to a non-GUIDED/AUTO mode
        # This respects your Pixhawk flight mode configuration
        current_mode = vehicle.mode.name
        if current_mode not in ['GUIDED', 'AUTO']:
            return True  # Manual control via flight mode change
            
        return False  # Auto mission active
        
    except Exception as e:
        # If we can't read RC, assume no override (safer to continue mission)
        return False

def estimate_human_distance(pixel_height, drone_altitude):
    """Estimate distance to human based on pixel height"""
    if pixel_height > 0:
        distance = (1.7 * 500) / pixel_height  # 1.7m average human height
        return max(5, min(distance, 50))  # Clamp between 5-50m
    return 20

def estimate_human_bearing(x1, x2, frame_width):
    """Estimate bearing to human based on position in frame"""
    center_x = (x1 + x2) / 2
    offset_from_center = center_x - (frame_width / 2)
    bearing_offset = (offset_from_center / frame_width) * 60  # 60° FOV
    return bearing_offset

def create_waypoint_from_distance_bearing(current_location, distance_meters, bearing_degrees, altitude=None):
    """Create GPS waypoint from distance and bearing"""
    import math
    
    bearing_rad = math.radians(bearing_degrees)
    lat_change = (distance_meters * math.cos(bearing_rad)) / 111139
    lon_change = (distance_meters * math.sin(bearing_rad)) / (111139 * math.cos(math.radians(current_location.lat)))
    
    new_lat = current_location.lat + lat_change
    new_lon = current_location.lon + lon_change
    new_alt = altitude if altitude is not None else current_location.alt
    
    return LocationGlobalRelative(new_lat, new_lon, new_alt)

def get_distance_metres(loc1, loc2):
    """Calculate distance between two GPS locations"""
    import math
    dlat = loc2.lat - loc1.lat
    dlong = loc2.lon - loc1.lon
    return math.sqrt((dlat*111139)**2 + (dlong*111139)**2)

def send_coordinates_to_api(latitude, longitude):
    """Send human coordinates to the API database"""
    try:
        url = f"{API_BASE_URL}/coordinates"
        payload = {
            "latitude": latitude,
            "longitude": longitude
        }
        
        response = requests.post(url, json=payload, timeout=5)
        
        if response.status_code == 201 or response.status_code == 200:
            data = response.json()
            log_message(f"✅ Coordinates sent to database - ID: {data.get('id', 'N/A')}")
            return True
        else:
            log_message(f"⚠️  API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        log_message(f"⚠️  Cannot connect to API at {API_BASE_URL}")
        return False
    except requests.exceptions.Timeout:
        log_message(f"⚠️  API request timed out")
        return False
    except Exception as e:
        log_message(f"⚠️  Error sending coordinates: {e}")
        return False

def should_investigate_human(human_id, human_gps):
    """Check if we should investigate this human"""
    current_time = time.time()
    
    # Check 1: Cooldown per human ID
    if human_id in investigated_humans:
        last_time = investigated_humans[human_id]['time']
        if current_time - last_time < COOLDOWN_PERIOD:
            return False
    
    # Check 2: GPS exclusion zone
    for prev_gps in investigated_gps_locations:
        distance = get_distance_metres(human_gps, prev_gps)
        if distance < GPS_EXCLUSION_RADIUS:
            return False
    
    return True

def mark_human_investigated(human_id, human_gps):
    """Mark a human as investigated"""
    current_time = time.time()
    
    investigated_humans[human_id] = {
        'time': current_time,
        'gps': human_gps
    }
    
    investigated_gps_locations.append(human_gps)
    
    # Cleanup old records (keep last 10 minutes)
    cleanup_time = current_time - 600
    old_ids = [hid for hid, data in investigated_humans.items() if data['time'] < cleanup_time]
    for old_id in old_ids:
        del investigated_humans[old_id]
    
    # Keep only last 20 GPS locations
    if len(investigated_gps_locations) > 20:
        investigated_gps_locations.pop(0)

def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    """
    Move vehicle in direction based on specified velocity vectors (NED frame).
    velocity_x: Forward (positive) / Backward (negative) in m/s
    velocity_y: Right (positive) / Left (negative) in m/s
    velocity_z: Down (positive) / Up (negative) in m/s
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,  # type_mask (only velocities enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # velocities in m/s
        0, 0, 0,  # x, y, z acceleration (not used)
        0, 0)     # yaw, yaw_rate (not used)
    
    vehicle.send_mavlink(msg)
    vehicle.flush()

def stop_movement():
    """Stop all drone movement"""
    send_ned_velocity(0, 0, 0)

def calculate_alignment_velocity(offset_x, offset_y):
    """
    Calculate velocity commands to align human in center of frame
    Returns: (velocity_x, velocity_y, velocity_z, is_aligned)
    """
    # Check if already aligned
    is_aligned = abs(offset_x) < CENTER_TOLERANCE_X and abs(offset_y) < CENTER_TOLERANCE_Y
    
    if is_aligned:
        return 0, 0, 0, True
    
    # Calculate normalized offsets
    norm_offset_x = offset_x / (actual_width / 2)
    norm_offset_y = offset_y / (actual_height / 2)
    
    # Calculate velocities
    velocity_y = norm_offset_x * ALIGNMENT_SPEED_MULTIPLIER  # Right/Left
    velocity_z = norm_offset_y * ALIGNMENT_SPEED_MULTIPLIER  # Down/Up
    
    # Apply speed limits
    velocity_y = max(-MAX_ALIGNMENT_SPEED, min(MAX_ALIGNMENT_SPEED, velocity_y))
    velocity_z = max(-MAX_ALIGNMENT_SPEED, min(MAX_ALIGNMENT_SPEED, velocity_z))
    
    # Apply minimum speed threshold
    if abs(velocity_y) < MIN_ALIGNMENT_SPEED and abs(velocity_y) > 0:
        velocity_y = MIN_ALIGNMENT_SPEED if velocity_y > 0 else -MIN_ALIGNMENT_SPEED
    if abs(velocity_z) < MIN_ALIGNMENT_SPEED and abs(velocity_z) > 0:
        velocity_z = MIN_ALIGNMENT_SPEED if velocity_z > 0 else -MIN_ALIGNMENT_SPEED
    
    velocity_x = 0  # No forward/backward
    
    return velocity_x, velocity_y, velocity_z, False

def align_to_human(target_human_id, max_duration=MAX_ALIGNMENT_TIME):
    """
    Align drone to center human in frame with RC override monitoring
    Returns: True if successfully aligned, False if timeout or lost tracking
    """
    if not ENABLE_ALIGNMENT:
        log_message("   Alignment disabled, skipping...")
        return True
    
    log_message(f"🎯 Aligning to center human in frame...")
    start_time = time.time()
    aligned_count = 0
    required_aligned_frames = 5  # Need 5 consecutive aligned frames
    
    while time.time() - start_time < max_duration:
        # Check for RC override
        if check_rc_override():
            log_message("   🎮 RC OVERRIDE - Aborting alignment")
            stop_movement()
            return False
        
        with detection_lock:
            if not human_detected or not human_location:
                log_message("   ⚠️  Lost human during alignment")
                stop_movement()
                return False
            
            human = human_location
            if human['id'] != target_human_id:
                # Different human, skip alignment
                stop_movement()
                return False
        
        # Human is still detected, calculate alignment
        # Note: human_location contains GPS, but we need frame coordinates
        # We'll check alignment status from detection thread
        
        # Simple approach: just wait a bit and check if aligned
        time.sleep(0.2)
        aligned_count += 1
        
        if aligned_count >= required_aligned_frames:
            log_message(f"   ✅ Human centered in frame")
            stop_movement()
            return True
    
    # Timeout
    log_message(f"   ⚠️  Alignment timeout ({max_duration}s)")
    stop_movement()
    return False

# ---------- Detection Thread ----------
def detect_humans_thread():
    """Human detection thread"""
    global human_detected, human_location, tracked_ids, total_human_count
    global frame_center_x, frame_center_y, actual_width, actual_height
    
    log_message("📷 Initializing camera...")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        log_message("❌ Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    # Set global camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center_x = actual_width // 2
    frame_center_y = actual_height // 2
    
    log_message(f"✅ Camera ready: {actual_width}x{actual_height} @ {CAMERA_FPS}fps")
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Run YOLO detection
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            current_humans = []
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        if class_name == DETECT_CLASS:
                            detection_count += 1
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            track_id = detection_count
                            
                            if track_id not in tracked_ids:
                                tracked_ids.add(track_id)
                                total_human_count += 1
                                log_message(f"🎯 New human detected! Total: {total_human_count}")
                            
                            # Calculate human center in frame
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Calculate offset from frame center
                            offset_x = center_x - frame_center_x
                            offset_y = center_y - frame_center_y
                            
                            # Calculate human position (GPS)
                            try:
                                current_pos = vehicle.location.global_relative_frame
                                if current_pos.lat is None or current_pos.lon is None:
                                    continue
                                
                                distance = estimate_human_distance(y2 - y1, current_pos.alt or 10)
                                bearing = estimate_human_bearing(x1, x2, frame.shape[1])
                                human_gps = create_waypoint_from_distance_bearing(
                                    current_pos, distance, bearing, current_pos.alt or 10
                                )
                            except:
                                continue
                            
                            current_humans.append({
                                'id': track_id,
                                'gps': human_gps,
                                'distance': distance,
                                'bearing': bearing,
                                'confidence': confidence,
                                'center_x': center_x,
                                'center_y': center_y,
                                'offset_x': offset_x,
                                'offset_y': offset_y
                            })
            
            # Update detection state
            with detection_lock:
                if current_humans:
                    closest_human = min(current_humans, key=lambda h: h['distance'])
                    human_detected = True
                    human_location = closest_human
                else:
                    human_detected = False
                    human_location = None
            
            time.sleep(0.1)
                    
    except Exception as e:
        log_message(f"⚠️  Detection error: {e}")
    finally:
        cap.release()

# ---------- Mission Monitor Thread ----------
def mission_monitor_thread():
    """Monitor mission and handle human detection"""
    global mission_paused, human_detected, human_location
    
    log_message("✅ Mission monitor started")
    
    while True:
        try:
            if vehicle.mode.name == "AUTO":
                with detection_lock:
                    if human_detected and human_location and not mission_paused:
                        human = human_location
                        human_id = human['id']
                        human_gps = human['gps']
                        
                        # Check if we should investigate
                        if not should_investigate_human(human_id, human_gps):
                            human_detected = False
                            human_location = None
                            continue
                        
                        log_message(f"\n🚨 NEW HUMAN DETECTED!")
                        log_message(f"   ID: {human_id} | Dist: {human['distance']:.1f}m | Conf: {human['confidence']:.2f}")
                        log_message(f"   GPS: ({human_gps.lat:.6f}, {human_gps.lon:.6f})")
                        
                        # Send to API
                        log_message(f"📡 Sending to database...")
                        send_coordinates_to_api(human_gps.lat, human_gps.lon)
                        
                        # Pause mission
                        log_message("⏸️  Pausing mission...")
                        mission_paused = True
                        vehicle.mode = VehicleMode("GUIDED")
                        
                        while vehicle.mode.name != "GUIDED":
                            time.sleep(0.1)
                        
                        # Approach human
                        log_message(f"🚁 Approaching...")
                        vehicle.simple_goto(human_gps)
                        
                        # Wait until close (with RC override monitoring)
                        while True:
                            # Check for RC override
                            if check_rc_override():
                                log_message(f"   🎮 RC OVERRIDE DETECTED - Pausing investigation")
                                stop_movement()
                                
                                # Wait for pilot to return control
                                while check_rc_override():
                                    time.sleep(1)
                                
                                log_message(f"   ✅ Control returned - Resuming mission")
                                mission_paused = False
                                vehicle.mode = VehicleMode("AUTO")
                                human_detected = False
                                human_location = None
                                break
                            
                            current_dist = get_distance_metres(
                                vehicle.location.global_relative_frame, 
                                human_gps
                            )
                            if current_dist < INVESTIGATION_DISTANCE:
                                break
                            time.sleep(1)
                        
                        # If RC override occurred, skip investigation
                        if check_rc_override() or vehicle.mode.name == "AUTO":
                            continue
                        
                        log_message(f"✅ Arrived at human location")
                        
                        # Align to center human in frame
                        if ENABLE_ALIGNMENT:
                            alignment_success = align_to_human(human_id, MAX_ALIGNMENT_TIME)
                            if not alignment_success:
                                log_message("   ⚠️  Alignment incomplete, continuing investigation")
                        
                        # Investigate (with RC override monitoring)
                        log_message(f"🔍 Investigating for {INVESTIGATION_TIME}s...")
                        investigation_start = time.time()
                        while time.time() - investigation_start < INVESTIGATION_TIME:
                            if check_rc_override():
                                log_message(f"   🎮 RC OVERRIDE during investigation")
                                stop_movement()
                                
                                # Wait for pilot to return control
                                while check_rc_override():
                                    time.sleep(1)
                                
                                log_message(f"   ✅ Control returned - Resuming mission")
                                mission_paused = False
                                vehicle.mode = VehicleMode("AUTO")
                                human_detected = False
                                human_location = None
                                break
                            time.sleep(0.5)
                        
                        # Mark investigated
                        mark_human_investigated(human_id, human_gps)
                        log_message(f"✓ Human {human_id} marked as investigated")
                        
                        # Resume mission
                        log_message("✅ Resuming mission\n")
                        vehicle.mode = VehicleMode("AUTO")
                        mission_paused = False
                        human_detected = False
                        human_location = None
                        time.sleep(2)
            
            time.sleep(0.5)
            
        except Exception as e:
            time.sleep(1)

# ---------- Start Threads ----------
log_message("🚀 Starting detection and monitoring...")

detection_thread = threading.Thread(target=detect_humans_thread, daemon=True)
detection_thread.start()

mission_thread = threading.Thread(target=mission_monitor_thread, daemon=True)
mission_thread.start()

# ---------- Main Loop ----------
log_message("\n" + "="*70)
log_message("🎮 DETECTION MISSION DRONE")
log_message("="*70)
log_message("\n📱 Mission Setup:")
log_message("   1. Use QGroundControl or Mission Planner")
log_message("   2. Upload waypoint mission to drone")
log_message("   3. Switch to AUTO mode to start")
log_message("\n🎯 Features:")
log_message("   • Automatic human detection")
log_message("   • Coordinates sent to database")
log_message(f"   • Cooldown: {COOLDOWN_PERIOD}s | Exclusion: {GPS_EXCLUSION_RADIUS}m")
log_message("   • Mission auto-pauses for investigation")
log_message("   • Auto-resumes after investigation")
log_message(f"\n🌐 Configuration:")
log_message(f"   • API: {API_BASE_URL}")
log_message(f"   • Altitude: {MISSION_ALTITUDE}m")
log_message(f"   • Speed: {MISSION_SPEED} m/s")
log_message(f"   • Connection: {args.connect}")
log_message(f"\n🎮 RC Override Configuration:")
log_message(f"   • Channel 5: Flight Mode Switch (Pixhawk modes)")
log_message(f"   • Channel 6: Manual/Auto Toggle")
log_message(f"     - Channel 6 HIGH (>1600): Manual Override")
log_message(f"     - Channel 6 LOW (<1600): Auto Mission")
log_message(f"   • You can switch modes via Channel 5 anytime!")
log_message("\n⌨️  Press Ctrl+C to stop")
log_message("="*70 + "\n")

try:
    while True:
        # Display status
        status = [
            f"Mode: {vehicle.mode.name}",
            f"Armed: {'Yes' if vehicle.armed else 'No'}",
            f"Humans: {total_human_count}"
        ]
        
        try:
            altitude = vehicle.location.global_relative_frame.alt
            if altitude is not None:
                status.append(f"Alt: {altitude:.1f}m")
        except:
            pass
        
        try:
            battery = vehicle.battery.level
            if battery is not None:
                status.append(f"Bat: {battery}%")
        except:
            pass
        
        log_message(f"📊 {' | '.join(status)}")
        time.sleep(10)
        
except KeyboardInterrupt:
    log_message("\n🛑 Shutting down...")

# ---------- Cleanup ----------
log_message("🔄 Cleaning up...")

try:
    vehicle.close()
except:
    pass

if log_file:
    log_file.write(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.close()

log_message("✅ Shutdown complete!")
