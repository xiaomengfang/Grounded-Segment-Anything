import os
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision
from torchvision.transforms import ToTensor

from groundingdino.util.inference import Model
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
start = time.time()
class GroundedEfficientSAM(nn.Module):
    def __init__(self, GROUNDING_DINO_CONFIG_PATH = None,
                 GROUNDING_DINO_CHECKPOINT_PATH=None,
                 EFFICIENT_SAM_CHECHPOINT_PATH=None,
                 ):
        super().__init__()

        self.dir = os.path.dirname(__file__)        
        if GROUNDING_DINO_CONFIG_PATH is None:
            GROUNDING_DINO_CONFIG_PATH = os.path.join(self.dir, "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        if GROUNDING_DINO_CHECKPOINT_PATH is None:
            GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(self.dir, "../groundingdino_swint_ogc.pth")
        if EFFICIENT_SAM_CHECHPOINT_PATH is None:
            EFFICIENT_SAM_CHECHPOINT_PATH = os.path.join(self.dir, "./efficientsam_s_gpu.jit")

        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        self.efficientsam = torch.jit.load(EFFICIENT_SAM_CHECHPOINT_PATH)
        self.BOX_THRESHOLD = 0.25
        self.TEXT_THRESHOLD = 0.25
        self.NMS_THRESHOLD = 0.8

    def efficient_sam_box_prompt_segment(self, image, pts_sampled, model):
        bbox = torch.reshape(torch.tensor(pts_sampled), [1, 1, 2, 2])
        bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = ToTensor()(image)

        predicted_logits, predicted_iou = model(
            img_tensor[None, ...].cuda(),
            bbox.cuda(),
            bbox_labels.cuda(),
        )
        predicted_logits = predicted_logits.cpu()
        all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
            ):
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]
        return selected_mask_using_predicted_iou
    
    def segment (self,image,CLASSES):
        
        detections = self.grounding_dino_model.predict_with_classes(
                                                    image=image,
                                                    classes=CLASSES,
                                                    box_threshold=self.BOX_THRESHOLD,
                                                    text_threshold=self.BOX_THRESHOLD
                                                )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # save the annotated grounding dino image
        cv2.imwrite(f"{self.dir}/groundingdino_annotated_image.jpg", annotated_frame)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")
        
        # collect segment results from EfficientSAM
        result_masks = []
        result_boxes = []
        for box in detections.xyxy:
            mask = self.efficient_sam_box_prompt_segment(image, box, self.efficientsam)
            # mask_img = torch.zeros(mask.shape[-2:])
            # mask_img[mask == True] = 1
            # cv2.imwrite("EfficientSAM/mask.jpg", mask_img.unsqueeze(dim=-1).numpy() * 255)
            result_masks.append(mask)
            result_boxes.append(box)
        detections.mask = np.array(result_masks)

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return result_masks, result_boxes, annotated_image
        # return mask, annotated_image

if __name__ == "__main__":

    dir = os.path.dirname(__file__)
    SOURCE_IMAGE_PATH = f"/home/epic/Projects/mobile_manipulation/data/raw_rgb_img_1.png"
    
    CLASSES = ["ball"]
    # CLASSES = ["bench"]
    image = cv2.imread(SOURCE_IMAGE_PATH)
    sam = GroundedEfficientSAM()
    # mask, annotated_image = sam.segment(image,CLASSES)
    masks, boxes, annotated_image = sam.segment(image,CLASSES)
    mask = masks[0]
    mask_img = torch.zeros(mask.shape[-2:])
    mask_img[mask == True] = 1
    cv2.imwrite(f"{dir}/mask12.jpg", mask_img.unsqueeze(dim=-1).numpy() * 255)

    cv2.imwrite(f"{dir}/gronded_efficient_sam_anontated_image12.jpg", annotated_image)
