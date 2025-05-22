import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.load import load_data
from utils.preprocessing import preprocess
from utils.split import split_data

def main():
    pass

#A mettre à la fin
if __name__ == "__main__":
    main()