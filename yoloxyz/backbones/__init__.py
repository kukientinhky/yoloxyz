import sys
import os 

yolov7_path = os.path.abspath(__file__).split('/')[-3:-1]
sys.path.append(os.path.join('/'.join(yolov7_path), 'yolov7'))
