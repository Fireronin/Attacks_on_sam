import numpy as np
import cv2
import torch
from typing import Optional, Tuple
import matplotlib as plt

def show_mask(mask, ax, sam, random_color=False,target_size=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30.0/255, 144.0/255, 255.0/255, 0.6])
   
    # mask = np.maximum(mask, np.zeros_like(mask))
    # mask = mask/np.max(mask)
    mask = mask > sam.mask_threshold
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if target_size:
        mask = cv2.resize(mask, (target_size, target_size))
    # print(np.min(mask))
    # print(np.max(mask))
    ax.imshow(mask)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def predict_torch(
    sam,
    image_embeddings,
    point_coords: Optional[torch.Tensor],
    point_labels: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor] = None,
    mask_input: Optional[torch.Tensor] = None,
    multimask_output: bool = True,
    return_logits: bool = False, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    if point_coords is not None:
        points = (point_coords, point_labels)
    else:
        points = None

    # Embed prompts
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=points,
        boxes=None,
        masks=None,
    )

    # Predict masks
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
    )

    # Upscale the masks to the original image resolution
    masks = sam.postprocess_masks(low_res_masks, (1024,1024), (256,256))

    if not return_logits:
        masks = masks > sam.mask_threshold

    return masks, iou_predictions, low_res_masks

def iou_float(mask1, mask2):
    intersection = torch.min(mask1, mask2)
    union = torch.max(mask2, mask1)
    iou_score = torch.sum(intersection) / torch.sum(union)
    
    return iou_score

def mse(mask1, mask2):
    return torch.mean((mask1 - mask2)**2)