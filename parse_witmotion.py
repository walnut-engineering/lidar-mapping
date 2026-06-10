import serial
import time

def parse_witmotion():
    try:
        with serial.Serial('/dev/ttyS1', 230400, timeout=0.1) as ser:
            print("Listening for WitMotion packets...")
            buffer = bytearray()
            start = time.time()
            while time.time() - start < 1.5:
                # read whatever is available
                if ser.in_waiting:
                    buffer.extend(ser.read(ser.in_waiting))
                else:
                    d = ser.read(11)
                    if d: buffer.extend(d)
                
                # Need at least 11 bytes to parse a packet
                while len(buffer) >= 11:
                    # Find packet start
                    if buffer[0] != 0x55:
                        buffer.pop(0)
                        continue
                    
                    # We have a candidate packet - check checksum
                    packet = buffer[:11]
                    checksum = sum(packet[:10]) & 0xFF
                    if checksum != packet[10]:
                        buffer.pop(0)
                        continue
                        
                    # Valid packet! Parse type
                    ptype = packet[1]
                    if ptype == 0x51: # Accel
                        ax = int.from_bytes(packet[2:4], 'little', signed=True) / 32768.0 * 16 * 9.81
                        ay = int.from_bytes(packet[4:6], 'little', signed=True) / 32768.0 * 16 * 9.81
                        az = int.from_bytes(packet[6:8], 'little', signed=True) / 32768.0 * 16 * 9.81
                        print(f"Accel: x={ax:5.2f} y={ay:5.2f} z={az:5.2f}")
                    elif ptype == 0x53: # Angle
                        r = int.from_bytes(packet[2:4], 'little', signed=True) / 32768.0 * 180
                        p = int.from_bytes(packet[4:6], 'little', signed=True) / 32768.0 * 180
                        y = int.from_bytes(packet[6:8], 'little', signed=True) / 32768.0 * 180
                        print(f"Angle: r={r:5.2f} p={p:5.2f} y={y:5.2f}")
                    
                    # Remove parsed packet
                    buffer = buffer[11:]
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == '__main__':
    parse_witmotion()
