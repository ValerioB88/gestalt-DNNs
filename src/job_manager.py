import argparse
import os
import time
from sty import ef, rs, fg, bg
from filelock import FileLock
import numpy as np
from datetime import datetime
gpu_lock_file = './gpu.txt'

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-command", type=str)
parser.add_argument("-gpus", metavar='N', type=int, nargs='+')
parser.add_argument("-max", type=int, default=1)
parser.add_argument("-reset", action='store_true')


PARAMS = vars(parser.parse_known_args()[0])

if PARAMS['reset']:
    with open(gpu_lock_file, 'w') as file:
        print(ef.inverse + f"Resetting GPU file.")
        exit(0)
        pass
message_said = False
lock = FileLock(gpu_lock_file + '.lock')
cmd = PARAMS['command']
print(ef.inverse + f"Executing command {cmd}" + rs.inverse)
gpus = PARAMS['gpus']
max_jobs = PARAMS['max']
print(ef.inverse + f"Requested any GPUs num {gpus}" + rs.inverse)
available_gpus = []
while not available_gpus:
    lock.acquire()
    if not os.path.exists(gpu_lock_file):
        with open(gpu_lock_file, 'w') as file:
            pass
    with open(gpu_lock_file, 'r') as file:
        g_locked = file.readlines()
    if not g_locked:
        gpu_locked = []
    else:
        gpu_locked = [int(g.split("\n")[0]) for g in g_locked]

    available_gpus = [g for g in gpus if g not in gpu_locked or (g in gpu_locked and np.sum(np.array(g) == gpu_locked) < max_jobs)]
    if not available_gpus:
        print(ef.inverse + f"GPUs {gpu_locked} are busy.. sleeping for 5 seconds (this message won't be repeated)" + rs.inverse) if not message_said else None
        message_said = True
        lock.release()
        time.sleep(5)

selected_gpu = available_gpus[0]
print(ef.inverse + f"Locked GPUs: {gpu_locked}, Free GPUs: {available_gpus}, selecting {selected_gpu}" + rs.inverse)
with open(gpu_lock_file, 'a') as file:
    file.write(f'{selected_gpu}\n')
lock.release()

try:
    session_name = datetime.now().strftime("%H%M%S%f")
    # whole_cmd = f'tmux new-session -d -s {session_name} && tmux send-keys -t {session_name} \'' + cmd + f' -use_device_num {selected_gpu}\' Enter'
    # print(whole_cmd) # this works but doesn't help, all it's run within a docker container and you can't have access to individual tmux sessions. Plus the process will seem as stopped and the gpus won't be released! What a shame!
    print(cmd + f' -use_device_num {selected_gpu}')
    os.system(cmd + f' -use_device_num {selected_gpu}')
    # os.system(cmd + f' -use_device_num {selected_gpu}')
except:
    print(ef.inverse + "Failed." + rs.inverse)

lock.acquire()
with open(gpu_lock_file, 'r') as file:
    g_locked = file.read()

gpu_locked = [int(i) for i in g_locked.split()]
gpu_locked.remove(selected_gpu)
print(ef.inverse + f"Freeing GPU {selected_gpu}, locked GPUs: {gpu_locked}" + rs.inverse)
with open(gpu_lock_file, 'w') as file:
    [file.write(f"{n}\n") for n in gpu_locked]

lock.release()