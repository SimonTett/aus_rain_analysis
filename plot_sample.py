# plot sample of rainrate for Sydney. Date chosen is when max rain occurs.
import pathlib

import numpy as np
import cartopy.crs as ccrs


import ausLib
import xarray
import matplotlib.pyplot as plt
base_dir = pathlib.Path.home() / "data/aus_rain_analysis/radar/"
ds=xarray.load_dataset(base_dir/'sample_data/54_20131102_rainrate.nc')
max_time= ds.rainrate.idxmax('time').assign_coords(longitude=ds.longitude,latitude=ds.latitude)
rng = np.sqrt(max_time.x.astype('float')**2+max_time.y.astype('float')**2)/1e3
rng =rng.assign_coords(longitude=ds.longitude,latitude=ds.latitude)
fig=plt.figure(num='sample_sydney',clear=True,figsize=(11,8))
ax=fig.add_subplot(111,projection=ccrs.PlateCarree())
max_time.dt.hour.plot(x='longitude',y='latitude',ax=ax)
rng.plot.contour(ax=ax,levels=np.linspace(0,150,16),colors='black',linestyles='solid',
                 x='longitude',y='latitude')
ax.coastlines()
ax.set_title("UTC Hour  of max rain on 2013-11-02")
fig.show()
fig.savefig('figures/sydney_max_hr.png')

## lets make a plot... thansk to chatgpt for example code.
import matplotlib.animation as animation
from matplotlib import widgets

rain = ds.rainrate.assign_coords(longitude=ds.longitude,latitude=ds.latitude)
# Load your xarray dataset
# Replace 'filename.nc' with the path to your dataset


# Define a function to update the plot for each frame of the animation
def update(frame):
    plt.clf()  # Clear the previous plot
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # Plot your data for the current frame
    # For example, assuming 'data_variable' is the variable you want to plot
    levels=[-1,0,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,250]
    r=rain.isel(time=frame)
    time = r.time.values
    r.plot(ax=ax, transform=ccrs.PlateCarree(),x='longitude',y='latitude', cmap='jet',levels=levels)
    rng.plot.contour(ax=ax, levels=np.arange(0, 8)*20, colors='black', linestyles='solid',
                     x='longitude', y='latitude')
    ax.coastlines()  # Add coastlines
    ax.set_title(f'Time: {time}')  # Set the title for the current frame
    print(f"done {time}")
# Create a figure
fig = plt.figure(figsize=(10, 5),num='animate_max_sydney',clear=True)

# Define the total number of frames (time steps) in the animation
num_frames = len(rain.time)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False, repeat=False)

# Save the animation
# Replace 'animation.mp4' with the desired filename and extension (e.g., 'animation.gif')
ani.save('animation.gif', writer='imageMagic', fps=2)  # Adjust fps (frames per second) as needed
# Add animation controls
play_pause_ax = fig.axes((0.1, 0.01, 0.8, 0.03))  # Define the position of the play/pause button
play_pause_button = widgets.Button(play_pause_ax, 'Play/Pause', hovercolor='lightgray')

# Define functions for play/pause button
def play_pause(event):
    if ani.event_source is None:
        ani.event_source = fig.canvas.new_timer(interval=100)  # Adjust interval as needed
        ani.event_source.add_callback(ani._step)
        ani.event_source.start()
    else:
        ani.event_source.remove()
        ani.event_source = None

play_pause_button.on_clicked(play_pause)

# Add a slider for frame selection
num_frames = rain.time.size
frame_slider_ax = plt.axes((0.1, 0.05, 0.8, 0.03))  # Define the position of the slider
frame_slider = widgets.Slider(frame_slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

# Define function for slider update
def update_slider(val):
    ani.frame_seq = ani.new_frame_seq()  # Reset frame sequence
    ani.frame_seq = [int(frame_slider.val)]  # Set current frame
    update(int(frame_slider.val))  # Update plot

frame_slider.on_changed(update_slider)
# Show the animation
plt.show()

