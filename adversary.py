from dotenv import load_dotenv
load_dotenv()

from util import predict_torch, display_image

import numpy as np
import torch
import cv2
from torch.autograd import Variable
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

EPOCHS = 25
LEARNING_RATE = 1
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

IMAGE_PATH = './person.jpg'
INPUT_POINTS = np.array([[120.0, 170.0]]) #([[120.0, 170.0],[120.0, 100.0]])
INPUT_LABELS = np.array([1.0]) #([1.0, 1.0])
INPUT_BOX = np.array([[41.0, 12.0, 236.0, 256.0]])

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
transform = ResizeLongestSide(sam.image_encoder.img_size)

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original_size = image.shape[-2:]
image = transform.apply_image(image)
image = torch.tensor(image, requires_grad=True, dtype=torch.float, device=device).permute(2, 0, 1).contiguous().unsqueeze(0)
original_image = image.clone().detach()

input_points_transformed = transform.apply_coords(INPUT_POINTS, original_size)
input_points_transformed = torch.tensor(input_points_transformed, dtype=torch.float, device=device, requires_grad=True).unsqueeze(0).detach()
input_labels_transformed = torch.tensor(INPUT_LABELS, dtype=torch.int, device=device).unsqueeze(0).detach()
input_box_transformed = transform.apply_boxes(INPUT_BOX, original_size)
input_box_transformed = torch.tensor(input_box_transformed, dtype=torch.float, device=device, requires_grad=True).detach()

display_image(image, points_labels=(input_points_transformed, input_labels_transformed), boxes=input_box_transformed, title="Input points", show_axis=True)

masks, scores = predict_torch(sam, original_image, point_coords=input_points_transformed, point_labels=input_labels_transformed, boxes=input_box_transformed, multimask_output=False, return_logits=True)
target_mask = masks.squeeze().detach()
score = scores.squeeze()

display_image(image, mask=target_mask, mask_threshold=sam.mask_threshold, points_labels=(input_points_transformed, input_labels_transformed), boxes=input_box_transformed, title=f"Score: {score.item():.3f}")

image = Variable(image, requires_grad=True)
optimizer = torch.optim.Adam([image], lr=LEARNING_RATE)

sam.eval()
for param in sam.parameters():
    param.requires_grad = False
    
def loss(image, target_mask, input_points, input_labels, input_box):
    masks, _ = predict_torch(sam, image, point_coords=input_points, point_labels=input_labels, boxes=input_box, multimask_output=False, return_logits=True)
    mask = masks.squeeze()
    return -torch.nn.functional.mse_loss(mask, target_mask), mask

# Gradient descent loop
for i in range(EPOCHS): 
    optimizer.zero_grad() 

    loss_value, mask = loss(image, target_mask, input_points_transformed, input_labels_transformed, input_box_transformed)
    loss_value.backward()

    optimizer.step()
    with torch.no_grad():
        image[image<0] = 0
        image[image>255] = 255

    display_image(image, mask=mask, mask_threshold=sam.mask_threshold, points_labels=(input_points_transformed, input_labels_transformed), boxes=input_box_transformed)

display_image(original_image, title='Original image')

display_image(image, title='Edited image')