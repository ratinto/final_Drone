#!/usr/bin/env python3
"""
Raspberry Pi Delivery Drone - Autonomous Package Delivery System
Fetches human coordinates from database and delivers packages
For real drone with USB MAVLink connection
"""

from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import argparse
import requests
import cv2
from pymavlink import mavutil
from ultralytics import YOLO

# ---------- Configuration ----------
# Connection Settings
DEFAULT_CONNECTION_STRING = "/dev/ttyACM0"  # USB MAVLink connection
DEFAULT_BAUD_RATE = 57600

# API Configuration
API_BASE_URL = "https://server-drone.vercel.app/api"

# RC Override Settings
# Channel 5: Flight modes (configured in Pixhawk) - STABILIZE, ALT_HOLD, LOITER, etc.
# Channel 6: Manual/Auto mission toggle (high = manual override, low = auto mission)
RC_CHANNEL_6_THRESHOLD = 1600  # Channel 6 value above which manual mode is active
RC_OVERRIDE_CHECK_INTERVAL = 0.5  # seconds - how often to check for RC override

# YOLO Model
MODEL_PATH = "best.pt"  # Your trained human detection model
CONFIDENCE_THRESHOLD = 0.4
DETECT_CLASS = "human"

# Camera Settings
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 20

# Delivery Settings
DELIVERY_ALTITUDE = 10  # meters
DELIVERY_SPEED = 5  # m/s
WAYPOINT_RADIUS = 2.0  # meters - how close to consider "arrived"
HOVER_TIME = 5  # seconds - time to hover at each delivery point
MAX_DELIVERIES = 10  # Maximum deliveries per mission

# Alignment Settings
ENABLE_ALIGNMENT = True  # Enable centering human in frame
CENTER_TOLERANCE_X = 40  # pixels - horizontal tolerance
CENTER_TOLERANCE_Y = 40  # pixels - vertical tolerance
MAX_ALIGNMENT_SPEED = 0.5  # m/s - maximum alignment speed
MIN_ALIGNMENT_SPEED = 0.1  # m/s - minimum alignment speed
ALIGNMENT_SPEED_MULTIPLIER = 1.5  # sensitivity of movement
MAX_ALIGNMENT_TIME = 10  # seconds - maximum time spent aligning

# Log File
LOG_FILE = "delivery_log.txt"

# ---------- Global Variables ----------
log_file = None
model = None
cap = None

# Camera properties (set after initialization)
frame_center_x = 0
frame_center_y = 0
actual_width = 0
actual_height = 0

# ---------- Setup Logging ----------
log_file = open(LOG_FILE, 'a')
log_file.write(f"\n{'='*70}\n")
log_file.write(f"Delivery mission started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"{'='*70}\n")
log_file.flush()

def log_message(message):
    """Log message to console and file"""
    print(message)
    if log_file:
        log_file.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
        log_file.flush()

# ---------- Parse Command Line Arguments ----------
parser = argparse.ArgumentParser(description='Raspberry Pi Delivery Drone')
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
                   default=DELIVERY_ALTITUDE,
                   help='Delivery altitude in meters (default: 10)')
parser.add_argument('--speed',
                   type=float,
                   default=DELIVERY_SPEED,
                   help='Delivery speed in m/s (default: 5)')
parser.add_argument('--max-deliveries',
                   type=int,
                   default=MAX_DELIVERIES,
                   help='Maximum number of deliveries (default: 10)')

args = parser.parse_args()
API_BASE_URL = args.api
DELIVERY_ALTITUDE = args.altitude
DELIVERY_SPEED = args.speed
MAX_DELIVERIES = args.max_deliveries

# ---------- Connect to Vehicle ----------
log_message(f"\n🔌 Connecting to vehicle on: {args.connect}")

try:
    log_message("⏳ Connecting...")
    vehicle = connect(args.connect, baud=args.baud, wait_ready=True, timeout=60)
    log_message("✅ Vehicle connected!")
    log_message(f"   Battery: {vehicle.battery.level}% | Mode: {vehicle.mode.name} | GPS: {vehicle.gps_0.fix_type}")
    
    # Set default speeds
    log_message(f"⚡ Setting default speed to {DELIVERY_SPEED} m/s...")
    vehicle.groundspeed = DELIVERY_SPEED
    vehicle.airspeed = DELIVERY_SPEED
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

# ---------- Initialize Camera ----------
log_message("📷 Initializing camera...")
try:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        log_message("❌ Failed to open camera")
        exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    # Set global camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center_x = actual_width // 2
    frame_center_y = actual_height // 2
    
    log_message(f"✅ Camera ready: {actual_width}x{actual_height} @ {CAMERA_FPS}fps")
except Exception as e:
    log_message(f"❌ Camera initialization failed: {e}")
    exit(1)

# ---------- Helper Functions ----------
def check_rc_override():
    """Check if pilot is taking manual control via RC transmitter
    
    Returns:
        True if manual override is active (either via Channel 6 or non-GUIDED mode)
        False if autonomous mission should continue
    """
    try:
        # Check RC channel 6 (Manual/Auto toggle)
        # High position (>1600) = Manual override
        # Low position (<1600) = Auto mission
        rc6 = vehicle.channels['6']
        
        if rc6 > RC_CHANNEL_6_THRESHOLD:
            return True  # Manual mode via Channel 6
        
        # Check if flight mode was changed via Channel 5 to a non-GUIDED mode
        # This respects your Pixhawk flight mode configuration
        current_mode = vehicle.mode.name
        if current_mode not in ['GUIDED', 'AUTO']:
            return True  # Manual control via flight mode change
            
        return False  # Auto mission active
        
    except Exception as e:
        # If we can't read RC, assume no override (safer to continue mission)
        return False

def detect_human_in_frame():
    """Detect human in current camera frame"""
    global cap, model, frame_center_x, frame_center_y
    
    if cap is None or not cap.isOpened():
        return False, 0, 0, 0, 0
    
    ret, frame = cap.read()
    if not ret:
        return False, 0, 0, 0, 0
    
    # Run YOLO detection
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                if class_name == DETECT_CLASS:
                    # Human detected!
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate human center in frame
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Calculate offset from frame center
                    offset_x = center_x - frame_center_x
                    offset_y = center_y - frame_center_y
                    
                    return True, center_x, center_y, offset_x, offset_y
    
    return False, 0, 0, 0, 0

def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    """Move vehicle in direction based on specified velocity vectors"""
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
    """Calculate velocity commands to align human in center of frame"""
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

def align_to_human(max_duration=MAX_ALIGNMENT_TIME):
    """Align drone to center human in frame with RC override monitoring"""
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
        
        # Detect human in frame
        human_found, center_x, center_y, offset_x, offset_y = detect_human_in_frame()
        
        if not human_found:
            log_message("   ⚠️  No human detected in frame")
            stop_movement()
            time.sleep(0.2)
            continue
        
        # Calculate alignment velocities
        vx, vy, vz, is_aligned = calculate_alignment_velocity(offset_x, offset_y)
        
        if is_aligned:
            aligned_count += 1
            if aligned_count >= required_aligned_frames:
                log_message(f"   ✅ Human centered in frame")
                stop_movement()
                return True
        else:
            aligned_count = 0
            # Send velocity commands
            send_ned_velocity(vx, vy, vz)
        
        time.sleep(0.1)
    
    # Timeout
    log_message(f"   ⚠️  Alignment timeout ({max_duration}s)")
    stop_movement()
    return False

# ---------- API Functions ----------
def get_unvisited_coordinates():
    """Fetch all unvisited coordinates from the database"""
    try:
        url = f"{API_BASE_URL}/coordinates/status/unvisited"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle different API response formats
            if isinstance(result, dict) and 'data' in result:
                coords = result['data']
            elif isinstance(result, list):
                coords = result
            else:
                log_message(f"⚠️  Unexpected API response format")
                return []
            
            log_message(f"📥 Retrieved {len(coords)} unvisited locations from database")
            return coords
        else:
            log_message(f"⚠️  API returned status {response.status_code}")
            return []
            
    except requests.exceptions.ConnectionError:
        log_message(f"⚠️  Cannot connect to API server at {API_BASE_URL}")
        return []
    except Exception as e:
        log_message(f"⚠️  Error fetching coordinates: {e}")
        return []

def mark_coordinate_visited(coord_id):
    """Mark a coordinate as visited in the database"""
    try:
        url = f"{API_BASE_URL}/coordinates/{coord_id}/visited"
        response = requests.patch(url, timeout=5)
        
        if response.status_code == 200:
            log_message(f"✅ Coordinate ID {coord_id} marked as visited")
            return True
        else:
            log_message(f"⚠️  Failed to mark visited - Status {response.status_code}")
            return False
            
    except Exception as e:
        log_message(f"⚠️  Error marking visited: {e}")
        return False

def mark_coordinate_delivered(coord_id):
    """Mark a coordinate as delivered in the database"""
    try:
        url = f"{API_BASE_URL}/coordinates/{coord_id}/delivered"
        response = requests.patch(url, timeout=5)
        
        if response.status_code == 200:
            log_message(f"✅ Coordinate ID {coord_id} marked as delivered")
            return True
        else:
            log_message(f"⚠️  Failed to mark delivered - Status {response.status_code}")
            return False
            
    except Exception as e:
        log_message(f"⚠️  Error marking delivered: {e}")
        return False

# ---------- Flight Functions ----------
def get_distance_metres(loc1, loc2):
    """Calculate distance between two GPS locations"""
    dlat = loc2.lat - loc1.lat
    dlong = loc2.lon - loc1.lon
    return math.sqrt((dlat*111139)**2 + (dlong*111139)**2)

def arm_and_takeoff(target_altitude):
    """Arms vehicle and fly to target_altitude"""
    log_message("🔧 Performing pre-arm checks...")
    
    while not vehicle.is_armable:
        log_message("   Waiting for vehicle to initialize...")
        time.sleep(1)
    
    log_message("✅ Vehicle is armable")
    
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    
    while not vehicle.armed:
        log_message("   Waiting for arming...")
        time.sleep(1)
    
    log_message("✅ Vehicle armed!")
    
    log_message(f"🚀 Taking off to {target_altitude}m...")
    vehicle.simple_takeoff(target_altitude)
    
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        if current_altitude is None:
            time.sleep(1)
            continue
            
        log_message(f"   Altitude: {current_altitude:.1f}m")
        
        if current_altitude >= target_altitude * 0.95:
            log_message("✅ Reached target altitude")
            break
        time.sleep(1)

def goto_location(latitude, longitude, altitude):
    """Fly to a specific GPS location with RC override monitoring"""
    target_location = LocationGlobalRelative(latitude, longitude, altitude)
    vehicle.simple_goto(target_location)
    
    return target_location

def wait_for_arrival(target, waypoint_radius=WAYPOINT_RADIUS):
    """Wait until arrived at target location, with RC override monitoring"""
    while True:
        # Check for RC override
        if check_rc_override():
            log_message("🎮 RC OVERRIDE DETECTED - Pilot taking manual control!")
            stop_movement()
            return False  # Abort current waypoint
        
        current_location = vehicle.location.global_relative_frame
        distance = get_distance_metres(current_location, target)
        
        if distance <= waypoint_radius:
            return True  # Arrived successfully
        
        time.sleep(2)

def deliver_package(coord_id, location_number, total_locations):
    """Perform delivery at current location"""
    log_message(f"\n📦 DELIVERY {location_number}/{total_locations}")
    log_message(f"   Coordinate ID: {coord_id}")
    log_message(f"   Hovering for {HOVER_TIME} seconds...")
    
    # Hover at location
    for i in range(HOVER_TIME):
        remaining = HOVER_TIME - i
        log_message(f"   ⏱️  {remaining}s remaining...")
        time.sleep(1)
    
    log_message(f"   ✅ Package delivered!")

# ---------- Main Delivery Mission ----------
def run_delivery_mission():
    """Main delivery mission logic"""
    log_message("\n" + "="*70)
    log_message("📦 STARTING DELIVERY MISSION")
    log_message("="*70)
    
    # Step 1: Fetch coordinates from database
    log_message("\n📡 Fetching delivery locations from database...")
    coordinates = get_unvisited_coordinates()
    
    if not coordinates:
        log_message("❌ No unvisited locations found. Mission aborted.")
        return False
    
    # Limit to MAX_DELIVERIES
    if len(coordinates) > MAX_DELIVERIES:
        log_message(f"⚠️  Found {len(coordinates)} locations, limiting to {MAX_DELIVERIES}")
        coordinates = coordinates[:MAX_DELIVERIES]
    
    total_deliveries = len(coordinates)
    log_message(f"\n📋 Delivery Plan: {total_deliveries} locations")
    for i, coord in enumerate(coordinates, 1):
        log_message(f"   {i}. ID {coord['id']}: ({coord['latitude']:.6f}, {coord['longitude']:.6f})")
    
    # Step 2: Arm and takeoff
    log_message("\n" + "="*70)
    arm_and_takeoff(DELIVERY_ALTITUDE)
    
    # Step 3: Visit each coordinate
    log_message("\n" + "="*70)
    log_message("🚁 STARTING DELIVERY ROUTE")
    log_message("="*70)
    
    successful_deliveries = 0
    
    for i, coord in enumerate(coordinates, 1):
        coord_id = coord['id']
        latitude = coord['latitude']
        longitude = coord['longitude']
        
        log_message(f"\n📍 Waypoint {i}/{total_deliveries}")
        log_message(f"   Target: ({latitude:.6f}, {longitude:.6f})")
        log_message(f"   Flying to location...")
        
        # Fly to location
        target = goto_location(latitude, longitude, DELIVERY_ALTITUDE)
        
        # Wait until arrived (with RC override monitoring)
        log_message(f"   Distance monitoring...")
        arrival_success = wait_for_arrival(target, WAYPOINT_RADIUS)
        
        if not arrival_success:
            log_message(f"   🎮 RC OVERRIDE DETECTED - Pausing mission")
            log_message(f"   Waiting for pilot to return control...")
            
            # Wait for pilot to return to GUIDED mode
            while check_rc_override():
                time.sleep(1)
            
            log_message(f"   ✅ Control returned - Resuming mission")
            continue  # Skip this delivery and move to next
        
        log_message(f"   ✅ Arrived at location!")
        
        # Mark as visited immediately upon arrival
        mark_coordinate_visited(coord_id)
        
        # Align to center human in frame
        log_message(f"\n🎯 Centering human for delivery...")
        if ENABLE_ALIGNMENT:
            alignment_success = align_to_human(MAX_ALIGNMENT_TIME)
            if not alignment_success:
                log_message("   ⚠️  Alignment incomplete, continuing delivery anyway")
        
        # Perform delivery
        deliver_package(coord_id, i, total_deliveries)
        
        # Mark as delivered after successful delivery
        if mark_coordinate_delivered(coord_id):
            successful_deliveries += 1
    
    # Step 4: Check if all deliveries complete
    log_message("\n" + "="*70)
    log_message("📊 DELIVERY SUMMARY")
    log_message("="*70)
    log_message(f"   Total Locations: {total_deliveries}")
    log_message(f"   Successful Deliveries: {successful_deliveries}")
    log_message(f"   Failed Deliveries: {total_deliveries - successful_deliveries}")
    
    if successful_deliveries == total_deliveries:
        log_message("   ✅ ALL DELIVERIES COMPLETED!")
        return True
    else:
        log_message("   ⚠️  Some deliveries failed")
        return False

def return_to_launch():
    """Return to launch location"""
    log_message("\n" + "="*70)
    log_message("🏠 RETURNING TO LAUNCH")
    log_message("="*70)
    
    vehicle.mode = VehicleMode("RTL")
    
    log_message("🚁 Flying back to home...")
    
    # Wait until landed
    while vehicle.armed:
        altitude = vehicle.location.global_relative_frame.alt
        if altitude is not None:
            log_message(f"   Altitude: {altitude:.1f}m")
        time.sleep(2)
    
    log_message("✅ Landed safely at home")

# ---------- Execute Mission ----------
try:
    log_message("\n" + "="*70)
    log_message("🚁 RASPBERRY PI AUTONOMOUS DELIVERY DRONE")
    log_message("="*70)
    log_message(f"\n🌐 Configuration:")
    log_message(f"   API URL: {API_BASE_URL}")
    log_message(f"   Max Deliveries: {MAX_DELIVERIES}")
    log_message(f"   Delivery Altitude: {DELIVERY_ALTITUDE}m")
    log_message(f"   Delivery Speed: {DELIVERY_SPEED} m/s")
    log_message(f"   Waypoint Radius: {WAYPOINT_RADIUS}m")
    log_message(f"   Alignment: {'Enabled' if ENABLE_ALIGNMENT else 'Disabled'}")
    log_message(f"   Connection: {args.connect}")
    log_message(f"\n🎮 RC Override Configuration:")
    log_message(f"   Channel 5: Flight Mode Switch (Pixhawk modes)")
    log_message(f"   Channel 6: Manual/Auto Toggle")
    log_message(f"     • Channel 6 HIGH (>1600): Manual Override")
    log_message(f"     • Channel 6 LOW (<1600): Auto Mission")
    log_message(f"   You can also switch to manual modes via Channel 5 anytime!")
    log_message("\n⏳ Starting mission in 3 seconds...")
    log_message("   Press Ctrl+C to abort")
    time.sleep(3)
    
    # Run the delivery mission
    mission_success = run_delivery_mission()
    
    # Return to launch
    return_to_launch()
    
    # Final status
    log_message("\n" + "="*70)
    if mission_success:
        log_message("✅ MISSION COMPLETED SUCCESSFULLY")
    else:
        log_message("⚠️  MISSION COMPLETED WITH WARNINGS")
    log_message("="*70)
    
except KeyboardInterrupt:
    log_message("\n�� Mission interrupted by user")
    log_message("�� Initiating emergency RTL...")
    vehicle.mode = VehicleMode("RTL")
    time.sleep(5)

except Exception as e:
    log_message(f"\n❌ Mission error: {e}")
    log_message("🚨 Initiating emergency RTL...")
    vehicle.mode = VehicleMode("RTL")
    time.sleep(5)

finally:
    # ---------- Cleanup ----------
    log_message("\n🔄 Cleaning up...")
    time.sleep(2)
    
    try:
        vehicle.close()
    except:
        pass
    
    # Release camera
    if cap is not None:
        try:
            cap.release()
        except:
            pass
    
    if log_file:
        log_file.write(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.close()
    
    log_message("✅ Shutdown complete!")
