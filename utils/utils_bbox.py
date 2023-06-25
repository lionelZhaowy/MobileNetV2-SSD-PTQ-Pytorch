import numpy as np
import torch
from torch import nn
from torchvision.ops import nms


class BBoxUtility(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def ssd_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        """从带灰条的图调整到原来的图"""
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)# 左边坐标
        box_maxes   = box_yx + (box_hw / 2.)# 右边坐标
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        return boxes

    def decode_boxes(self, mbox_loc, anchors, variances):
        """
        解码先验框(anchor boxes)和预测的边界框(bounding boxes)的偏移量,以得到预测框的实际位置
        将相对坐标转换为预测框的绝对坐标，使得我们能够将预测框直接应用于图像中
        """
        # 获得先验框的宽与高
        anchor_width     = anchors[:, 2] - anchors[:, 0]
        anchor_height    = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x  = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y  = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 预测框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        decode_bbox_center_y += anchor_center_y

        # 预测框的宽与高的求取,对预测框宽、高的偏移量进行指数函数操作,计算出预测框的实际宽、高
        decode_bbox_width   = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width   *= anchor_width
        decode_bbox_height  = torch.exp(mbox_loc[:, 3] * variances[1])
        decode_bbox_height  *= anchor_height

        # 获取预测框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 预测框的左上角与右下角进行堆叠
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                 decode_bbox_ymin[:, None],
                                 decode_bbox_xmax[:, None],
                                 decode_bbox_ymax[:, None]), dim=-1)
        # 防止超出0与1,确保预测框的坐标在图像范围内，避免越界问题
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox

    def decode_box(self, predictions, anchors, image_shape, input_shape, letterbox_image, variances = [0.1, 0.2], nms_iou = 0.3, confidence = 0.5):
        """对检测结果进行后处理，将预测结果解码，使用非极大抑制（NMS）筛选出置信度较高的目标框，
        并将其转换为左上角坐标和右下角坐标的形式，然后返回筛选后的结果"""
        #---------------------------------------------------#
        #   :4是回归预测结果
        #---------------------------------------------------#
        mbox_loc        = predictions[0]
        #---------------------------------------------------#
        #   获得种类的置信度
        #---------------------------------------------------#
        mbox_conf       = nn.Softmax(-1)(predictions[1])

        results = []
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(len(mbox_loc)):
            results.append([])
            #--------------------------------#
            #   利用回归结果对先验框进行解码
            #   variances:用于解码框坐标的方差
            #--------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors, variances)

            # 第0类为背景类 因此从第1类开始 
            for c in range(1, self.num_classes):
                #--------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                #--------------------------------#
                c_confs     = mbox_conf[i, :, c]
                c_confs_m   = c_confs > confidence
                if len(c_confs[c_confs_m]) > 0:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    #-----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    #-----------------------------------------#
                    good_boxes = boxes_to_process[keep]
                    confs      = confs_to_process[keep][:, None]
                    labels     = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    #-----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    #-----------------------------------------#
                    c_pred     = torch.cat((good_boxes, labels, confs), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)
            # 调整结果形式 从带灰条的图调整到原来的图
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results
