import os
import shutil
import random
import sys
import argparse

def create_test_set():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples",type=int,help="Number of images in the test set")
    args = parser.parse_args()
    data_dir = os.path.join(os.getcwd(),'data','celebA')
    if not os.path.exists(data_dir):
        print('{} does not exist'.format(data_dir))
        sys.exit()
    data_files = (os.listdir(data_dir)) # List of all files in the data directory
    target_dir = os.path.join(os.getcwd(),'test_set')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else: # Clean it
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)

    files = [os.path.join(data_dir,f) for f in data_files]
    test_files = random.sample(files,args.num_samples)
    for tFiles in test_files:
        shutil.copy(tFiles,target_dir)


if __name__ == "__main__":
    create_test_set()
