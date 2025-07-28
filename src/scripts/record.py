import subprocess
import time

start_time = time.time()
print(f"Recording start system time: {start_time}")

with open("start_time.txt", "w") as f:
    f.write(str(start_time))

subprocess.call(["k4arecorder", "-l", "30", "timestamptest.mkv"])
