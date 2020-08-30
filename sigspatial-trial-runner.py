import subprocess
import sys
import os
import time

if len(sys.argv) > 1:
    trial_category = str(sys.argv[1])
else:
    trial_category = str(input('Trial category (example: 3): '))

if len(sys.argv) > 2:
    delay_time = int(sys.argv[2]) * 60
else:
    delay_time = int(input('Delay time before running in minutes (example: 180): ')) * 60

if len(sys.argv) > 3:
    rs = str(sys.argv[3])
else:
    rs = str(input('Override random state (type a positive integer to do so)? '))

time.sleep(delay_time)

for path in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config-files', 'sigspatial-trials')):
    if path.startswith(trial_category):
        s = r'start "" py simulation.py sigspatial-trials\{} {}'.format(path[:-4], rs)
        print(s)
        subprocess.call(s, shell=True)
        time.sleep(60)
