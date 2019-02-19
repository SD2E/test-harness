print("I am the perovskite test harness script")

import argparse
import sklearn
import tensorflow

parser = argparse.ArgumentParser()
parser.add_argument('commit_hash', help='First 7 characters of versioned data commit hash')

if __name__ == '__main__':
     with open('test_me.txt', 'w') as fout:
          fout.write('hello')

     args = parser.parse_args()

     print(args.commit_hash)
     print("I finished the perovskite test harness script")
