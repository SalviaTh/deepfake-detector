import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # If no target layer specified, try to find the last conv layer for EfficientNet
        if target_layer_name is None:
            # For EfficientNet, the last conv layer is usually in the backbone._conv_head
            self.target_layer = self.model.backbone._conv_head
        else:
            # Find the layer by name (simplified logic)
            self.target_layer = dict(self.model.named_modules())[target_layer_name]
            
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations by pooled gradients
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        # Normalize heatmap
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.detach().cpu().numpy()
        
        # Prediction info
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        
        return heatmap, target_class, probs

    @staticmethod
    def overlay(img_bgr, heatmap):
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on image
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        return overlay

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
