import cv2
import math
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import datetime
import scipy.stats as stats


# Import cleaned tracking data and information about fish
df_fish75 = pd.read_csv(r'F:\biodi_project\analysis_dlc\video_analysis_network_3points\cleaned_data\2023-07-27-181909_ID26_exp3-n3-speed12-n6-speed6_HAB_fish75_interpolated-cutoff005.csv')

# Calculate distance between two points
n_frames = np.arange(df_fish75.shape[0])
distance_pixels = []


for frame in n_frames:
    if frame == 0:
        distance = 0
    else:
        x_t0 = df_fish75['head_x'][frame - 1]
        y_t0 = df_fish75['head_y'][frame - 1]

        x_t1 = df_fish75['head_x'][frame]
        y_t1 = df_fish75['head_y'][frame]

        distance = math.sqrt((x_t1 - x_t0) ** 2 + (y_t1 - y_t0) ** 2)

    distance_pixels.append(distance)

# Plot distance over time
plt.plot(n_frames, distance_pixels)
plt.ylim(0, 50)

# Now I would like to make the same plot as before animated, so that at each frame of the animation the corresponding distance is plotted

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Create a function that plots the distance at each frame
def animate(i):
    data = distance_pixels[:i]
    ax.clear()
    ax.plot(n_frames[:i], data)
    ax.set_ylim(0, 50)
    ax.set_xlim(0, len(n_frames))
    ax.set_title('Distance over time')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance (pixels)')

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=range(0, len(distance_pixels)), interval=0.2)

# Save the animation
#ani.save('distance_over_time.mp4')

# Show the animation
plt.show()





