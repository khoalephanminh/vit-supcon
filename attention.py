import torch
from scipy.ndimage import rotate, zoom

def get_attention_map_3d(model, img, anomaly):
    model.eval()
    with torch.no_grad():
        img = img.unsqueeze(0)  # Add batch dimension, shape: (1, 1, 120, 160, 160)
         # (batch_size, channels, frames, height, width)
        anomaly = anomaly.unsqueeze(0)
        _ = model(img, anomaly)
        att_mat = model.attention_weights
        att_mat = torch.stack(att_mat).squeeze(1)  # Shape: (num_layers, num_heads, num_patches, num_patches)
        att_mat = torch.mean(att_mat, dim=1)  # Average the attention weights across all heads.
        
        residual_att = torch.eye(att_mat.size(-1)).to(att_mat.device)
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)
        # (num_layers, num_patches, num_patches)
        
        joint_attentions = torch.zeros_like(aug_att_mat).to(att_mat.device)
        joint_attentions[0] = aug_att_mat[0] # Get attention weight at the first layer
        
        for n in range(1, aug_att_mat.size(0)): # Loop throught layers to compute the overall attention relationships across all layers
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            
        v = joint_attentions[-1]  # Shape: (num_patches, num_patches) # (1401,1401) get weight attention at the last layer
        grid_size = 10 # int(np.round(np.cbrt(aug_att_mat.size(-1))))
        mask = v[0, 401:].reshape(grid_size, grid_size, grid_size).detach().cpu().numpy()
        zoom_factors = (img.size(2) / grid_size, img.size(3) / grid_size, img.size(4) / grid_size)
        resized_mask = zoom(mask, zoom_factors, order=1)
        result = (resized_mask / resized_mask.max()) * img.squeeze().cpu().numpy()
    return result

def plot_attention_map_3d(original_img, att_map, anomaly, slice_index=39, label=None, prediction=None, save_path=None):
    fig, (ax1, ax2,ax3) = plt.subplots(ncols=3, figsize=(16, 16))
    ax1.set_title(f'Label: {label}, Prediction: {prediction}')
    ax1.axis('off')
    ax2.set_title(f'Attention map')
    ax2.axis('off')
    ax3.set_title(f'Anomaly map')
    ax3.axis('off')
    
    _ = ax1.imshow(original_img[0, slice_index, :, :], cmap='gray')
    _ = ax2.imshow(att_map[slice_index, :, :], cmap='jet')
    _ = ax3.imshow(np.squeeze(anomaly,axis=0), cmap='turbo')
    if save_path:
        plt.savefig(save_path)
    plt.show()