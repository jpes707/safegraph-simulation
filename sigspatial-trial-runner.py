import subprocess
import sys
import os

if len(sys.argv) > 1:
    trial_category = str(sys.argv[1])
else:
    trial_category = str(input('Trial category (example: 3): '))

for path in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config-files', 'sigspatial-trials')):
    if path.startswith(trial_category):
        subprocess.call(r'start "" py simulation.py sigspatial-trials\{}'.format(path[:-4]), shell=True)
