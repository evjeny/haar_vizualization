import argparse
import time

import cv2

import utils


parser = argparse.ArgumentParser(description="Haar vizualization")
parser.add_argument("--cascade", type=str, help="path to cascade")
parser.add_argument("--image", type=str, help="path to image")
parser.add_argument("--scale", type=float, default=1.0, help="scale of image")
parser.add_argument("--k", type=int, default=5, help="top K classifier's activations")
parser.add_argument("--output", type=str, default="output.avi", help="output video name")
parser.add_argument("--output_tps", type=float, default=1.0, help="time per stage of output video")
parser.add_argument("--output_width", type=int, default=-1, help="frame width of output video")
parser.add_argument("--output_height", type=int, default=-1, help="frame height of output video")
args = parser.parse_args()

with open(args.cascade, "r") as f:
    xml = f.read()
stages, features, width, height = utils.parse_cascade(xml)

image = cv2.imread(args.image, 0)
image_height, image_width = image.shape[:2]

new_image_height = int(image_height * args.scale)
new_image_width = int(image_width * args.scale)

image_scaled = cv2.resize(image, (new_image_width, new_image_height), interpolation=cv2.INTER_NEAREST)

t0 = time.time()
marked_images = utils.get_stage_images(image_scaled, stages, features, height, width, args.k)
t1 = time.time()
print(t1 - t0, "s")

if args.output_width == -1 or args.output_height == -1:
    args.output_width = new_image_width
    args.output_height = new_image_height

utils.save_video(args.output, marked_images, args.output_tps, args.output_width, args.output_height)
print("saved output to", args.output)

