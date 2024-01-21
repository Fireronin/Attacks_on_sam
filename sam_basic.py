# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from torch.autograd import Variable
from typing import Optional, Tuple
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
def show_mask(mask, ax, random_color=False,target_size=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # resize the mask to target_size (if provided) (H,W) -> (target_size, target_size)
    if target_size:
        mask_image = cv2.resize(mask_image, (target_size, target_size))
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


# %% [markdown]
# ## Example image

# %%
image = cv2.imread('./person.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %%
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()

# %%
#sam_checkpoint = "sam_vit_h_4b8939.pth"
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# %%
predictor.set_image(image)

input_point = np.array([[120, 170],[120,100]])
input_label = np.array([1,1])

# %%
plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  

# %%
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
    return_logits=True,
)

# %%
masks.shape  # (number_of_masks) x H x W

# %%
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
  

#%%

orginal_mask = np.copy(masks[0])


# %%
transformed = predictor.transform.apply_image(image)
image_torch = torch.tensor(transformed, requires_grad=True, dtype=torch.float32,device=device)
image_torch = image_torch.permute(2, 0, 1)
image_torch = image_torch.unsqueeze(0)
image_torch = Variable(image_torch, requires_grad=True)


original_mask = torch.from_numpy(orginal_mask).float().to(device)


optimizer = torch.optim.Adam([image_torch], lr=0.1)

input_point_torch = torch.from_numpy(input_point).float()*4
input_label_torch = torch.from_numpy(input_label).float()
input_point_torch = input_point_torch.unsqueeze(0)
input_label_torch = input_label_torch.unsqueeze(0)
input_point_torch = input_point_torch.to(device)
input_label_torch = input_label_torch.to(device)

#sam.requires_grad_(True)
sam.eval()
# frrze the model
for param in sam.parameters():
    param.requires_grad = False

def predict_torch(
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

def iou(mask1, mask2):
    # mask1 = mask1 > sam.mask_threshold
    # mask2 = mask2 > sam.mask_threshold
    # print(mask1.shape)
    # print(mask2.shape)
    intersection = torch.min(mask1, mask2)
    union = torch.max(mask2, mask1)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score

def mse(mask1, mask2):
    return torch.mean((mask1 - mask2)**2)

def mask_reduction(orginal_mask, mask):
    # where orginal_mask is less than 0 ignore it
    # otherwise compute average of mask 
    # return the average
    mask= last_mask
    tmp = original_mask > sam.mask_threshold
    tmp = tmp.float()
    mask = mask * tmp
    mask = mask[mask>0]
    # 
    return torch.mean(mask).nan_to_num(0.0)
        

last_mask = None

def loss(image, original_mask, input_point, input_label):
    global last_mask
    image_embeddings = sam.image_encoder(image)
    masks, scores, logits = predict_torch(image_embeddings, point_coords=input_point, point_labels=input_label, multimask_output=False, return_logits=True, )
    last_mask = masks
    print(masks.shape)
    #return 1 - iou(original_mask, masks)
    return iou(original_mask, masks)


#%%
# Gradient descent loop
for i in range(5):  # Number of iterations
    optimizer.zero_grad()  # Zero out the gradients
    loss_value = loss(image_torch, original_mask, input_point_torch, input_label_torch)  # Compute the loss
    print(f"Loss: {loss_value}")
    loss_value.backward(retain_graph=True)  # Compute the gradients
    optimizer.step()  # Update the image tensor
    last_mask_numpy = last_mask.detach().cpu().numpy()

    # resize the mask to 1024,1024 using torch.nn.functional.interpolate
    # (256, 256)
    last_mask_numpy = torch.nn.functional.interpolate(torch.from_numpy(last_mask_numpy), size=(1024,1024), mode='bilinear')
    last_mask_numpy = last_mask_numpy>0
    # copy image_tensor to image_numpy but before that detach and cpu and numpy
    #image_numpy 
    image_numpy = image_torch.detach().cpu().numpy().squeeze().transpose(1,2,0)/255
    for i, mask in enumerate(last_mask_numpy):
        plt.figure(figsize=(10,10))
        plt.imshow(image_numpy)
        show_mask(mask, plt.gca())
        show_points(input_point*4, input_label, plt.gca())
        plt.axis('off')
        plt.show()  

# %%
# image_torch.shape  torch.Size([1, 3, 1024, 1024])
# visualize the image
# visualize the mask
# last_mask to numpy

# %%
last_mask_numpy = last_mask.detach().cpu().numpy()

# resize the mask to 1024,1024 using torch.nn.functional.interpolate
# (256, 256)
last_mask_numpy = torch.nn.functional.interpolate(torch.from_numpy(last_mask_numpy), size=(1024,1024), mode='bilinear')
last_mask_numpy = last_mask_numpy>0
# copy image_tensor to image_numpy but before that detach and cpu and numpy
#image_numpy 
image_numpy = image_torch.detach().cpu().numpy().squeeze().transpose(1,2,0)/255
for i, mask in enumerate(last_mask_numpy):
    plt.figure(figsize=(10,10))
    plt.imshow(image_numpy)
    show_mask(mask, plt.gca())
    show_points(input_point*4, input_label, plt.gca())
    plt.axis('off')
    plt.show()  
# %%
mask= last_mask
tmp = original_mask > sam.mask_threshold
tmp = tmp.float()
mask = mask * tmp
mask = mask[mask>0]
if mask.numel() == 0:  # Check if the tensor is empty
    result = 0
else:
    result = torch.mean(mask)

# %%
image_embeddings = sam.image_encoder(image_torch)
masks, scores, logits = predict_torch(image_embeddings, point_coords=input_point_torch, point_labels=input_label_torch, multimask_output=False, return_logits=True, )
masks = masks.cpu().detach().numpy()
scores = scores.cpu().detach().numpy()
# %%
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    # switch channels
    image_torch_fixed = image_torch.permute(0, 2, 3, 1)
    plt.imshow(image_torch_fixed.cpu().detach().numpy()[0]/255)
    #plt.imshow(image)
    show_mask(mask[0], plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score[0]:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
# %%


