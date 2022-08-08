import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from bin import train
from util import config

parser = argparse.ArgumentParser(description="")
parser.add_argument("--config", type=str, default="config/LGAImers_test.yaml")

args = parser.parse_args()

cfg = config.load_cfg_from_cfg_file(args.config)
train(args.config)

