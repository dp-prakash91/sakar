import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
from pymodbus.client import ModbusTcpClient

# PLC connection parameters

num_cameras = 6   # Number of Video Capture devices to check

# Create a client to connect to the PLC
client = ModbusTcpClient(host= '192.168.1.11' , port = 502 )
# Create a client to connect to the PLC
if client.connect():
    print("Connected to Modbus server")
else:
    print("Failed to connect to Modbus server")
    exit()

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

def send_trigger(person_detected):
    """Send Trigger signal based on person detection."""
    if person_detected:
        client.write_register(0,1)
        print("Human Presence Detected")
    else:
        client.write_register(0,0)
        print("Human Absence Detected")
    print(f"\033[91mPerson detected: {person_detected}\033[0m")

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
model = YOLO(r'C:\Users\sakar\Desktop\HDS_v1.0.0-main\HDS_v1.0.0-main\yolo11n_openvino_model')

previous_person_detected = False

# Connect to the PLC


while True:
    if not frame_queue.empty():
        combined_frame = frame_queue.get()
        total_height, total_width = combined_frame.shape[:2]

        # Run object detection
        results = model.track(
            source=combined_frame,
            imgsz=[total_height, total_width]
        )

        # Check if a person is detected
        person_detected = False

        # Draw bounding boxes and other annotations
        annotated_frame = results[0].plot()

        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls[0])] == 'person':
                    person_detected = True

        # Send GPIO Signal based on person detection change
        if person_detected != previous_person_detected:
            send_trigger(person_detected)
            previous_person_detected = person_detected

        # Display the combined frame with bounding boxes
        cv2.imshow("Combined Frame", annotated_frame)
        print(f"Total pixel dimensions: {total_width}x{total_height}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Disconnect from the PLC
        
        break

# Release all resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()