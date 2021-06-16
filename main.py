import json
import argparse
from run_train import train
from run_test import test
from prepare_data import prep_train_data, prep_test_data
from stat_utils import report_stat

parser = argparse.ArgumentParser()

parser.add_argument('-trp', '--train_path', default='./train_data/')
parser.add_argument('-tsp', '--test_path', default='./test_data/')
parser.add_argument('--train', action='store_true', default='false')
parser.add_argument('--test', action='store_true', default='true')
parser.add_argument('--train_model', default="DeepPavlov/bert-base-cased-conversational")
#parser.add_argument('--test_model', default="DeepPavlov/bert-base-cased-conversational")
parser.add_argument('--test_model', default="./GRADE1/")
parser.add_argument('--level', default=8)

def main():
    args = parser.parse_args()

    args.train = True
    args.test = False

    args.level = 8
    print(f"Using Classification level {args.level}")

    if args.level not in [2,5,8]:
        print("Select a valid level!")
        exit(1)



    if args.train:
        print("Train mode")
        datapath = prep_train_data(args.train_path)
        trained_model = train(datapath, args.train_model, args.level)


    if args.test:
        print("Test mode")
        #datapath = prep_test_data(args.test_path, None)
        datapath = prep_test_data(args.test_path, "./test_data/GRADE5.pkl")
        result = test(datapath, args.test_model, args.level, model=trained_model)
        session_results = report_stat(result)

    return


main()