#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2022 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import copy
import time
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2
import numpy as np
import rospy
import torch
import torch.backends.cudnn as cudnn
from cv_bridge import CvBridge, CvBridgeError
from models.experimental import attempt_load
from numpy import random
from sensor_msgs.msg import Image
from utils.datasets import LoadImages, LoadStreams, letterbox
from utils.general import (apply_classifier, check_img_size, check_imshow,
                           check_requirements, increment_path,
                           non_max_suppression, scale_coords, set_logging,
                           strip_optimizer, xyxy2xywh)
from utils.plots import plot_one_box
from utils.torch_utils import (TracedModel, load_classifier, select_device,
                               time_synchronized)


class Yolov7:
    def __init__(self):
        weights = rospy.get_param('~weights', 'yolov7.pt')
        size = rospy.get_param('~size', 640)
        self.conf_thres = rospy.get_param('~conf_thres', 0.25)
        self.iou_thres = rospy.get_param('~iou_thres', 0.45)
        device = rospy.get_param('~device', '0')
        self.view_img = rospy.get_param('~view_img', True)
        self.classes = rospy.get_param('~classes', None)
        self.agnostic_nms = rospy.get_param('~agnostic_nms', False)
        self.augment = rospy.get_param('~augment', False)
        no_trace = rospy.get_param('~no_trace', False)
        # parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
        # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
        # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        # parser.add_argument('--view-img', action='store_true', help='display results')
        # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        # parser.add_argument('--augment', action='store_true', help='augmented inference')
        # parser.add_argument('--update', action='store_true', help='update all models')
        # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        # parser.add_argument('--name', default='exp', help='save results to project/name')
        # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        # parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.size = check_img_size(size, s=self.stride)  # check img_size

        if not no_trace:
            self.model = TracedModel(self.model, self.device, self.size)

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()


        self.__cv_bridge = CvBridge()
        self.image_pub = rospy.Publisher("~visualization", Image, queue_size=1)
        rospy.Subscriber("~image", Image, self.detect, queue_size=1)


    def detect(self, msg):
        # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

        # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Set Dataloader
        # vid_path, vid_writer = None, None
        # if webcam:
        #     view_img = check_imshow()
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=self.size, stride=self.stride)
        # else:
        # dataset = LoadImages(source, img_size=self.size, stride=self.stride)

        try:
            cv_image = self.__cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr('Converting Image Error. ' + str(e))
            return
        img0 = cv_image.copy()
        img = letterbox(img0, self.size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        im0s = None

        with torch.no_grad():
            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.size, self.size).to(self.device).type_as(next(self.model.parameters())))  # run once
            old_img_w = old_img_h = self.size
            old_img_b = 1

            t0 = time.time()
            # for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                # else:
                # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                s, im0 = '', img0

                # p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if self.view_img:
                    self.image_pub.publish(self.__cv_bridge.cv2_to_imgmsg(im0, 'bgr8'))
                    # cv2.imshow(str(p), im0)
                    # cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #         print(f" The image with the result is saved in: {save_path}")
                #     else:  # 'video' or 'stream'
                #         if vid_path != save_path:  # new video
                #             vid_path = save_path
                #             if isinstance(vid_writer, cv2.VideoWriter):
                #                 vid_writer.release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             else:  # stream
                #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                #                 save_path += '.mp4'
                #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vid_writer.write(im0)

            # if save_txt or save_img:
            #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                #print(f"Results saved to {save_dir}{s}")

            print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    rospy.init_node('yolov7')
    node = Yolov7()
    rospy.spin()

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov7.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()
