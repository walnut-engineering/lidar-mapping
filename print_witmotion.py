import serial
import time

try:
    with serial.Serial('/dev/ttyS1', 115200, timeout=1.0) as ser:
        print("Listening on /dev/ttyS1 at 115200 baud...")
        buffer = b""
        start = time.time()
        while time.time() - start < 1.0:
            bytes_read = ser.read(11)
            if bytes_read:
                print("115200 Received:", " ".join(f"{b:02X}" for b in bytes_read))
    with serial.Serial('/dev/ttyS1', 230400, timeout=1.0) as ser:
        print("Listening on /dev/ttyS1 at 230400 baud...")
        buffer = b""
        start = time.time()
        while time.time() - start < 1.0:
            bytes_read = ser.read(11)
            if bytes_read:
                print("230400 Received:", " ".join(f"{b:02X}" for b in bytes_read))
except Exception as e:
    print(f"Error: {e}")
