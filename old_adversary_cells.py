def mask_reduction(orginal_mask, mask):
    # where orginal_mask is less than 0 ignore it
    # otherwise compute average of mask 
    # return the average
    mask= last_mask
    tmp = mask > sam.mask_threshold
    tmp = tmp.float()
    mask = mask * tmp
    mask = mask[mask>0]
    # 
    return torch.mean(mask).nan_to_num(0.0)

global_mask_numpy = global_mask.detach().cpu().numpy()

# resize the mask to 1024,1024 using torch.nn.functional.interpolate
global_mask_numpy = torch.nn.functional.interpolate(torch.from_numpy(global_mask_numpy), size=(1024,1024), mode='bilinear')
global_mask_numpy = global_mask_numpy>0

image_numpy = image.detach().cpu().numpy().squeeze().transpose(1,2,0)/255

plt.figure(figsize=(10,10))
plt.imshow(image_numpy)
show_mask(global_mask_numpy, plt.gca())
show_points(INPUT_POINTS*4, INPUT_LABELS, plt.gca())
plt.axis('off')
plt.show()

mask = global_mask
mask = mask * (mask > sam.mask_threshold).float()
mask = mask[mask>0]
if mask.numel() == 0:  # Check if the tensor is empty
    result = 0
else:
    result = torch.mean(mask)

image_embeddings = sam.image_encoder(image)
masks, scores, logits = predict_torch(image_embeddings, point_coords=input_point_torch, point_labels=input_label_torch, multimask_output=False, return_logits=True, )
mask = masks[0].cpu().detach().numpy()
score = scores[0].cpu().detach().numpy()

plt.figure(figsize=(10,10))
# switch channels
image_torch_fixed = image.permute(0, 2, 3, 1)
plt.imshow(image_torch_fixed.cpu().detach().numpy()[0]/255)
#plt.imshow(image)
show_mask(mask[0], plt.gca())
show_points(INPUT_POINTS, INPUT_LABELS, plt.gca())
plt.title(f"Score: {score[0]:.3f}", fontsize=18)
plt.axis('off')
plt.show()  