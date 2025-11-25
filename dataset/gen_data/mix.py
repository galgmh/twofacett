import os
import json
import argparse
import nltk
import random
from tqdm import tqdm

def read_json(path):
    with open(path,"r", encoding="utf8") as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


