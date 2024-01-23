from dotenv import load_dotenv
load_dotenv()

from util import show_points, show_mask, show_box, predict_torch, iou_float

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from torch.autograd import Variable
from segment_anything import sam_model_registry, SamPredictor

EPOCHS = 3
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

IMAGE_PATH = './person.jpg'
INPUT_POINTS = np.array([[120, 170],[120,100]])
INPUT_LABELS = np.array([1,1])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(INPUT_POINTS, INPUT_LABELS, plt.gca())
plt.title('Input points')
plt.axis('on')
plt.show()

original_image = image

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=INPUT_POINTS,
    point_labels=INPUT_LABELS,
    multimask_output=False,
    return_logits=True,
)
correct_mask = np.copy(masks[0])
score = scores[0]

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(correct_mask, plt.gca(), sam)
show_points(INPUT_POINTS, INPUT_LABELS, plt.gca())
plt.title(f"Score: {score:.3f}", fontsize=18)
plt.axis('off')
plt.show()

image = predictor.transform.apply_image(image)
image = torch.tensor(image, requires_grad=True, dtype=torch.float32,device=device).permute(2, 0, 1).unsqueeze(0)
image = Variable(image, requires_grad=True)

correct_mask = torch.from_numpy(correct_mask).float().to(device)

optimizer = torch.optim.Adam([image], lr=0.1)

input_point_torch = torch.from_numpy(INPUT_POINTS).float()*4
input_label_torch = torch.from_numpy(INPUT_LABELS).float()
input_point_torch = input_point_torch.unsqueeze(0).to(device)
input_label_torch = input_label_torch.unsqueeze(0).to(device)

sam.eval()
for param in sam.parameters():
    param.requires_grad = False
    
def loss(image, original_mask, input_point, input_label):
    image_embeddings = sam.image_encoder(image)
    masks, scores, logits = predict_torch(sam, image_embeddings, point_coords=input_point, point_labels=input_label, multimask_output=False, return_logits=True)
    mask = masks.squeeze()
    print(mask.shape)
    print(original_mask.shape)
    return -torch.nn.functional.mse_loss(mask, original_mask), mask

# Gradient descent loop
for i in range(EPOCHS):  # Number of iterations
    optimizer.zero_grad()  # Zero out the gradients
    loss_value, mask = loss(image, correct_mask, input_point_torch, input_label_torch)  # Compute the loss
    print(f"Loss: {loss_value}")
    loss_value.backward(retain_graph=True)  # Compute the gradients
    optimizer.step()  # Update the image tensor

    # resize the mask to 1024,1024 using torch.nn.functional.interpolate
    mask = torch.nn.functional.interpolate(mask.reshape(1, 1, mask.shape[-2], mask.shape[-1]), size=(1024,1024), mode='bilinear').squeeze().detach().cpu().numpy()
    # mask = mask > 0

    image_numpy = image.detach().cpu().numpy().squeeze().transpose(1,2,0)/255
    image_numpy = np.maximum(image_numpy, np.zeros_like(image_numpy))
    image_numpy = np.minimum(image_numpy, np.ones_like(image_numpy))

    plt.figure(figsize=(10,10))
    plt.imshow(image_numpy)
    show_mask(mask, plt.gca(), sam)
    show_points(INPUT_POINTS*4, INPUT_LABELS, plt.gca())
    plt.axis('off')
    plt.show()

plt.figure(figsize=(10,10))
plt.imshow(original_image)
plt.title('Original image')
plt.axis('off')
plt.show()

image_numpy = image.detach().cpu().numpy().squeeze().transpose(1,2,0)/255
image_numpy = np.maximum(image_numpy, np.zeros_like(image_numpy))
image_numpy = np.minimum(image_numpy, np.ones_like(image_numpy))
plt.figure(figsize=(10,10))
plt.imshow(image_numpy)
plt.title('Edited image')
plt.axis('off')
plt.show()