import sys
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_directory, '..'))

print(os.listdir())

from config import labels

print(labels)