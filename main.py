import os
import argparse
from simpleRNN import simplernn
from IMDbExample import train as trainIMDB
from GenerateText import train as trainGT
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"


def test_imbd():
    trainIMDB.train_test()


def test_generate_text():
    trainGT.train()


parser = argparse.ArgumentParser()
parser.add_argument("--test", type=int, default=0,
                    help="Choose to test text generate model == 0 or IMDB model == 1")
args = parser.parse_args()

if __name__ == "__main__":
    if args.test == 0:
        test_generate_text()
    else:
        test_imbd()
