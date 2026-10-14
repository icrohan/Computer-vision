import cv2
import imutils
import threading
import serial
import time

# Global variables
position = "Center"
lock = threading.Lock()  # Thread synchronization

def send_object():
    """Detect faces and update the position."""
    global position
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(1)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break
            frame = imutils.resize(frame, width=1000)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            with lock:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    frame_center_x = frame.shape[1] // 2
                    face_center_x = x + w // 2

                    if face_center_x < frame_center_x - 50:
                        position = "Left"
                    elif face_center_x > frame_center_x + 50:
                        position = "Right"
                    else:
                        position = "Center"

                    cv2.putText(frame, position, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def send_data():
    """Send position data to the serial port."""
    global position
    ser = serial.Serial('COM7', 115200)
    time.sleep(2)  # Wait for the serial connection to initialize

    try:
        while True:
            with lock:
                current_position = position

            if current_position == "Left":
                ser.write(b'L\n')  # Send 'L' with newline
            elif current_position == "Right":
                ser.write(b'R\n')  # Send 'R' with newline
            else:
                ser.write(b'C\n')  # Send 'C' with newline

            # Debug: Read response from Arduino
            if ser.in_waiting > 0:
                data = ser.readline().decode().strip()
                print("Received:", data)

    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
    finally:
        ser.close()

# Create threads
thread1 = threading.Thread(target=send_object, daemon=True)
thread2 = threading.Thread(target=send_data, daemon=True)

# Start threads
thread1.start()
thread2.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)  # Let threads run in the background
except KeyboardInterrupt:    print("Exiting program.")
