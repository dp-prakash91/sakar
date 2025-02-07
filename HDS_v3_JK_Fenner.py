import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import logging
from datetime import datetime
import os
import time
import psutil
from pymodbus.client import ModbusTcpClient

# __________________________________________
# PLC connection parameters
# Set up Modbus client
client = ModbusTcpClient('192.168.1.200', port=502)  # replace with your Modbus server IP and port

num_cameras = 6  # Number of Video Capture devices to check
# __________________________________________
# Create a client to connect to the PLC
if client.connect():
    print("Connected to Modbus server")
else:
    print("Failed to connect to Modbus server")
    exit()
# __________________________________________
def check_and_reconnect():
    """Check the connection and reconnect if needed."""
    if not client.is_open():
        print("Modbus client disconnected, attempting to reconnect.")
        if not client.connect():
            print("Failed to reconnect to Modbus server.")
            return False
    return True
# __________________________________________
# Initialize logging
log_file = "human_detection_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")
# __________________________________________
# Create a folder for screenshots
screenshot_folder = "screenshots"
os.makedirs(screenshot_folder, exist_ok=True)
# __________________________________________
def write_holding_register(address, value):
    response = client.write_register(address, value)
    if not response.isError():
        print(f"Successfully wrote {value} to holding register at address {address}")
    else:
        print(f"Error writing to holding register at address {address}")
# __________________________________________
def find_working_cameras():
    """Find working cameras on the system."""
    working_cameras = []
    for device in range(num_cameras):
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, _ = cap.read()
            if ret:
                working_cameras.append(device)
            cap.release()
    return working_cameras
# __________________________________________
def capture_frames(caps, frame_queue):
    """Capture frames from all cameras and put them in a queue."""
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        if frames:
            combined_frame = np.hstack(frames)
            frame_queue.put(combined_frame)
# __________________________________________
def log_detection(person_detected, frame):
    """Log the detection event and save a screenshot if a person is detected."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if person_detected:
        # Get CPU temperatures
        cpu_temperatures = get_cpu_temperatures()

        # Log in the desired format
        log_message = f"{timestamp} Human Presence Detected"
        for temp in cpu_temperatures:
            log_message += f" {temp}°C"

        logging.info(log_message)

        # Save a screenshot with the timestamp in the filename
        screenshot_path = os.path.join(screenshot_folder, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved: {screenshot_path}")
# __________________________________________
def send_trigger(person_detected):
    try:
        """Send Trigger signal based on person detection."""
        if person_detected:
            DB = 1
            print("Human Presence Detected")
        else:
            DB = 0
            print("Human Absence Detected")

        # Ensure connection to Modbus client
        if not check_and_reconnect():
            print("Modbus connection failed. Skipping trigger.")
            return

        # Try writing to holding register with retry mechanism
        if not write_holding_register(0, DB):
            print("Failed to write to Modbus register. Trigger not sent.")
    except Exception as e:
        print(f"Error in send_trigger: {e}")
       
# __________________________________________
def get_cpu_temperatures():
    """Get the temperatures of all CPU cores."""
    try:
        temps = psutil.sensors_temperatures()
        cpu_temperatures = []
        if "coretemp" in temps:
            for sensor in temps["coretemp"]:
                if sensor.label.startswith("Core"):
                    cpu_temperatures.append(sensor.current)
        else:
            print("CPU temperature sensor not found.")
            cpu_temperatures = [
                "N/A"
            ] * 4  # Placeholder for missing data if cores aren't found
        return cpu_temperatures
    except Exception as e:
        print(f"Error reading temperature: {e}")
        return ["N/A"] * 4  # Placeholder for error case
# __________________________________________
working_cameras_list = find_working_cameras()
print(f"Working cameras: {working_cameras_list}")

# Initialize video capture for each working camera
caps = [cv2.VideoCapture(device) for device in working_cameras_list]

# Create a queue to hold frames
frame_queue = queue.Queue(maxsize=1)

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames, args=(caps, frame_queue))
capture_thread.daemon = True
capture_thread.start()

# model = YOLO('yolo11n.pt')
model = YOLO('../yolo11n_openvino_model')

previous_person_detected = False
# __________________________________________
while True:
    if not frame_queue.empty():
        combined_frame = frame_queue.get()
        total_height, total_width = combined_frame.shape[:2]

        # Run object detection
        results = model.track(source=combined_frame, imgsz=[total_height, total_width])

        # Check if a person is detected
        person_detected = False

        # Create a blank image to draw only person detections
        human_only_frame = combined_frame.copy()

        # Loop through the detections to find a "person" class
        for result in results:
            for box in result.boxes:
                # Check if the detected class is "person"
                if model.names[int(box.cls[0])] == "person":
                    person_detected = True
                    # Draw the bounding box and label on the human-only frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(human_only_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = "Person"
                    cv2.putText(
                        human_only_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            if person_detected:
                write_holding_register(0, 1)
                log_detection(person_detected, human_only_frame)
            else:
                write_holding_register(0, 0)
        # Display the frame with only human detections
        #cv2.imshow("Human Detection Only", human_only_frame)
    # __________________________________________
    if cv2.waitKey(1) & 0xFF == ord("q"):
        client.close()
        break
# __________________________________________
# Release all resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
