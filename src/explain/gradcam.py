"""
Grad-CAM heatmaps: show WHERE the classifier thinks the defect is.

Uses `pytorch-grad-cam`. Produces an overlay heatmap for the inference UI and
for qualitative figures in the report.

Usage (as a function called by the backend):
    heatmap = gradcam_overlay(model, image_tensor, target_layer)

Owner: Member 3
"""


def gradcam_overlay(model, image_tensor, target_layer):
    raise NotImplementedError("Member 3: wrap pytorch-grad-cam, return overlay image")
