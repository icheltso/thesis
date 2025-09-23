# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:39:17 2025

@author: ichel
"""

import os

import cv2
import numpy as np
import imageio

from PIL import Image, ImageDraw, ImageFont

import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pdf2image import convert_from_path

'Set base path for saving'
save_xtra = os.path.join("SIMULATION","TOY","1D")
images_graph = []

T_burn = 10000
tau_start = np.log10(0.6)
tau_end = np.log10(0.01)
tau_vals = np.logspace(tau_start,tau_end,12)
burn_in_vals = T_burn / tau_vals

def smooth_ease(alpha):
    """Easing function: smooth step (ease-in-out)"""
    return alpha ** 2 * (3 - 2 * alpha)

for i in range(0, 12):  # Assuming filenames are graph1.pdf, graph2.pdf, etc.
    #save_dist_gif = "1d_cart_dist_tau_" + str(i) + ".pdf"
    save_dist_gif = "ESS_" + str(i) + ".pdf"
    fig_path_gif = os.path.join(save_xtra,save_dist_gif)
    
    if not os.path.exists(fig_path_gif):
        print(f"File not found: {fig_path_gif}")
        continue  # Skip missing files

    try:
        img = convert_from_path(fig_path_gif)[0]  # Convert first page
        images_graph.append(img)
    except Exception as e:
        print(f"Error converting {fig_path_gif}: {e}")
        
        
if images_graph:
    base_width, base_height = images_graph[0].size  # Use first image's size
    
    new_width = base_width + 300  # Add extra space for text

    #images_graph = [img.resize((base_width, base_height), Image.LANCZOS) for img in images_graph]

    #images_graph[0].save("animation.gif", save_all=True, append_images=images_graph[1:], duration=500, loop=0)
    
    # Convert PIL images to OpenCV format (NumPy arrays)
    #images_cv = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images_graph]
    
    images_cv = []
    for i, img in enumerate(images_graph):
        img = img.resize((base_width, base_height), Image.LANCZOS)

        # Convert to NumPy array
        img_np = np.array(img)

        # Add padding (white space) to the left for text
        padded_img = np.ones((base_height, new_width, 3), dtype=np.uint8) * 255
        padded_img[:, 300:, :] = img_np  # Shift the image right

        # Convert back to PIL for text overlay
        img_pil = Image.fromarray(padded_img)
        draw = ImageDraw.Draw(img_pil)

        # Choose a font (ensure 'arial.ttf' is available, or use default)
        try:
            font = ImageFont.truetype("arial.ttf", 28)  # Adjust size as needed
        except IOError:
            font = ImageFont.load_default()


        # Text to display
        text1 = "T_burn = 10000"  # Fixed text at the top
        text2 = f"dt = {tau_vals[i]:.3f}"  # Tau value
        text3 = f"k_burn = {burn_in_vals[i]:.2e}"  # Burn-in in scientific notation
        
        # Get text sizes
        text_height = font.getbbox(text1)[3] - font.getbbox(text1)[1]
        text_spacing = 10  # Spacing between lines

        # Position text in the middle left
        text_x = 20
        text_y_center = base_height // 2
        draw.text((text_x, text_y_center - text_height - text_spacing), text1, font=font, fill=(0, 0, 0))  # Top text
        draw.text((text_x, text_y_center), text2, font=font, fill=(0, 0, 0))  # Middle text
        draw.text((text_x, text_y_center + text_height + text_spacing), text3, font=font, fill=(0, 0, 0))  # Bottom text

        # Convert back to OpenCV format
        images_cv.append(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))

    
    
    
    # Generate smooth transitions with pauses
    smooth_frames = []
    num_static_frames = 15  # Pause on each frame for 15 frames
    num_interpolated = 20  # Smooth transition over 20 frames
    
    for j in range(len(images_cv) - 1):
        img1, img2 = images_cv[j], images_cv[j + 1]
        
        # Add static frames to "pause" on img1
        smooth_frames.extend([img1] * num_static_frames)
        
        for k in range(num_interpolated + 1):
            alpha = smooth_ease(k / float(num_interpolated + 1))  # Use easing function
            blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
            smooth_frames.append(blended)
            
    # Add pause on the last image
    smooth_frames.extend([images_cv[-1]] * num_static_frames)
    
    # Define video output parameters
    #output_filename = "smooth_dist.mp4"
    output_filename = "ESS_bar.mp4"
    frame_height, frame_width, _ = smooth_frames[0].shape
    fps = 20  # Adjust FPS for smoothness

    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Write frames to video
    for frame in smooth_frames:
        video_writer.write(frame)  # Write each frame

    video_writer.release()  # Finalize the video
    print(f"Video saved as {output_filename}")

    # Convert frames back to RGB (for GIF)
    #smooth_frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in smooth_frames]

    # Save as GIF
    #imageio.mimsave("smooth_animation.gif", smooth_frames_rgb, duration=0.05)
    
    
else:
    print("No images to save.")

#images_graph[0].save("animation.gif", save_all=True, append_images=images_graph[1:], duration=500, loop=0)