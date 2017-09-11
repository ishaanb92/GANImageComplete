import os
import shutil
import random
import sys

def create_test_set():
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
    test_files = random.sample(files,10)
    for tFiles in test_files:
        shutil.copy(tFiles,target_dir)


if __name__ == "__main__":
    create_test_set()
