import os
import sys

if __name__ == '__main__':
    _seed = int(sys.argv[1])
    os.system("CUDA_VISIBLE_DEVICES=0,1 python test_DDPG-segement.py %d 0 50000" % _seed)
    os.system("CUDA_VISIBLE_DEVICES=0,1 python test_DDPG-segement.py %d 50000 100000" % _seed)
    os.system("CUDA_VISIBLE_DEVICES=0,1 python test_DDPG-segement.py %d 100000 150000" % _seed)
    os.system("CUDA_VISIBLE_DEVICES=0,1 python test_DDPG-segement.py %d 150000 200000" % _seed)
