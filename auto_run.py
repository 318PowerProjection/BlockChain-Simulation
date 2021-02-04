import os
import time
import sys
from pathlib import Path
from params import max_episode, save_interval


if __name__ == '__main__':
    _seed = int(sys.argv[1])
    os.system("rm -rf ./Output/Model/seed=%d" % _seed)
    os.system("CUDA_VISIBLE_DEVICES=0,1 python simulation_DDPG.py %d" % _seed)
    # os.system("CUDA_VISIBLE_DEVICES=0,1 python test_DDPG.py %d" % _seed)
    while True:
        time.sleep(1)
        actor_model = Path(('./Output/Model/seed=%d/actor_%d.pth' % (_seed, max_episode)))
        if actor_model.is_file():
            os.system("CUDA_VISIBLE_DEVICES=0,1 python test_DDPG.py %d" % _seed)
            break
