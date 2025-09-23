# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:39:17 2025

@author: ichel
"""

import os

import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pdf2image import convert_from_path

'Set base path for saving'
save_xtra = os.path.join("SIMULATION","TOY","1D")
images_graph = []

for i in range(0, 12):  # Assuming filenames are graph1.pdf, graph2.pdf, etc.
    save_dist_gif = "1d_cart_dist_tau_" + str(i) + ".pdf"
    fig_path_gif = os.path.join(save_xtra,save_dist_gif)
    
    img = convert_from_path(fig_path_gif)[0]  # Convert first page
    images_graph.append(img)
    
images_graph[0].save("animation.gif", save_all=True, append_images=images_graph[1:], duration=500, loop=0)