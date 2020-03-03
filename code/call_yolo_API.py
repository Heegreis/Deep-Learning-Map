import argparse
import torch
from yolov3_API_minUseage import detect


if __name__ == '__main__':
    source = 'stream.txt'
    # with torch.no_grad():
    #     detect()
    with torch.no_grad():
        detect_gen = detect(source)
        for x in detect_gen:
            print(x)

