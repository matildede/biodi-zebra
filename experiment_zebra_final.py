import copy
import pandas as pd
import scipy
import numpy as np
from psychopy import visual, event, core, tools, data, gui, monitors
from pathlib import Path
import os

# Define and set monitor properties
PIXELS_MONITOR = [1920, 1080] # [width, height] of the monitor in pixels
myMon = monitors.Monitor('asus')
myMon.setWidth(53.06)  # Set the width of the monitor in cm
myMon.setSizePix(PIXELS_MONITOR) # Set the size of the monitor in pixels

#Metadata dictionary
data_dict = {
    'experiment_metadata': {
                "experiment_name": "biodi",
                "experimental_group": "exp2-n6-speed3-12",
                "session": 44,
                "experimenter": "Matilde",
                "screen1_side": "left",
                "screen1_stimulus": 'flock',
                "screen1_number_dots": 3,
                "screen1_speed": 6.0,
                "screen2_side": "right",
                "screen2_stimulus": 'flock',
                "screen2_number_dots": 6,
                "screen2_speed": 6.0,
                "animal_birth": '21/06/2023',
                "animal_age": '7wpf',
                "animal_genotype": 'AB',
                "comments": None},

    'stimuli_parameters': {
                'pause_duration': 1800,  # 1800 seconds = 30 min
                'stimuli_duration': 300,  # 300 seconds = 5 min
                'num_chambers': 3,
                'size_dots_cm': 0.15 #RANGE ANIMALS 0.8-1.8 mm
    }
}

#open GUI for metadata
dlg = gui.DlgFromDict(data_dict['experiment_metadata'], title="Experiment metadata", sortKeys=False)
if not dlg.OK:
    core.quit()

#add date info after the dialog has gone
data_dict['experiment_metadata']['date'] = data.getDateStr(format="%Y-%m-%d-%H%M")

# Create a DataFrame using the dictionary
metadata_df = pd.concat([pd.DataFrame(data_dict['experiment_metadata'], index=[0]),
                         pd.DataFrame(data_dict['stimuli_parameters'], index=[0])], axis=1)

# Save DataFrame to CSV file
name_folder = data_dict['experiment_metadata']['date'][:10]
path = Path(r'\\cimec-storage5\acn_lab\shared_acn\Matilde\zebrafish\biodi_experiment\experiment_data\video')
filename = data_dict['experiment_metadata']['date'] + "_" + \
           'ID' + str(data_dict['experiment_metadata']['session']) + "_" + \
           data_dict['experiment_metadata']['experiment_name'] + "_" + \
           data_dict['experiment_metadata']['experimental_group'] + '.csv'

new_path = path / data_dict['experiment_metadata']['date'][:10] / ('ID' + str(data_dict['experiment_metadata']['session'])) / 'metadata'
tracking_path_folder = path / data_dict['experiment_metadata']['date'][:10] / ('ID' + str(data_dict['experiment_metadata']['session'])) / 'tracking' # create a new folder for bonsai video files
new_path.mkdir(exist_ok=True, parents=True)
tracking_path_folder.mkdir(exist_ok=True, parents=True)

if data_dict['experiment_metadata']['experiment_name'] != "":
    metadata_df.to_csv(new_path / filename, index=False)

# Import stimuli dataframe
# Set directory
path = r"C:\Users\matilde.perrino\Documents\GitHub\biodi-zebra\stimuli"

df_screen1 = pd.read_csv(path + f"\df_{data_dict['experiment_metadata']['screen1_stimulus']}_"
                       f"{data_dict['experiment_metadata']['screen1_number_dots']}_"
                       f"speed{data_dict['experiment_metadata']['screen1_speed']}.csv")

df_screen2 = pd.read_csv(path + f"\df_{data_dict['experiment_metadata']['screen2_stimulus']}_"
                       f"{data_dict['experiment_metadata']['screen2_number_dots']}_"
                       f"speed{data_dict['experiment_metadata']['screen2_speed']}.csv")

# dataframes
df_screen1_tot = [copy.copy(df_screen1) for i in range(data_dict['stimuli_parameters']['num_chambers'])]
df_screen2_tot = [copy.copy(df_screen2) for i in range(data_dict['stimuli_parameters']['num_chambers'])]
df_tot = [df_screen1_tot, df_screen2_tot]


# 1. Draw a grid of red lines to adjust the setup at the beginning of the experiment.
# Coordinates and variables for drawing grid lines to align the setup at the beginning of the experiment
WIDTH_LINES_cm = 0.5 # width of the lines in cm
LENGTH_LINES_cm = 11 # length of the lines in cm
NUMBER_OF_LINES = 4 # number of lines to draw
COORDINATES_LINES_cm = [15.25, 6.4]  # x and y coordinates of the first line (rectangle psychopy)

# Coordinates and variables for setting cells (areas where stimuli will be displayed)
NUM_CHAMBERS = 3
WIDTH_CELL_cm = 6
x_RANGE_CELL_cm = [np.array([15.5, 21.5]) + (WIDTH_CELL_cm + WIDTH_LINES_cm) * i for i in range(NUM_CHAMBERS)] # x coordinates
y_RANGE_CELL_cm = [0.9, 6.9] # y coordinates

# STEP. 1 Define and transform points coordinates from cm to px
# Lines coordinates
coordinates_lines_px = []
for line in range(NUMBER_OF_LINES):
    x_point = tools.monitorunittools.cm2pix(COORDINATES_LINES_cm[0] + (WIDTH_CELL_cm + WIDTH_LINES_cm) * line, myMon)
    y_point = tools.monitorunittools.cm2pix(COORDINATES_LINES_cm[1], myMon)
    coordinates_lines_px.append([x_point, y_point])

width_lines_px = tools.monitorunittools.cm2pix(WIDTH_LINES_cm, myMon)  # width of the lines in px
length_lines_px = tools.monitorunittools.cm2pix(LENGTH_LINES_cm, myMon) # length of the lines in px

# Cell range coordinates (areas where stimuli will be displayed)
x_RANGE_CELL_px = []
for i in range(NUM_CHAMBERS):
    x_range_points_px = [tools.monitorunittools.cm2pix(x_RANGE_CELL_cm[i][j], myMon) for j in range(len(x_RANGE_CELL_cm[i]))]
    x_RANGE_CELL_px.append(x_range_points_px)

y_RANGE_CELL_px = [tools.monitorunittools.cm2pix(y_RANGE_CELL_cm[j], myMon) for j in range(len(y_RANGE_CELL_cm))]

def translate(original_value, scala_min, scala_max):
    """
    This function traslate a normalized value on a scale 0-1 in a desired scale
    within the range from scala_min to scala_max

    :param original_value: value to be translated
    :param scala_min: minimum value of the desired scale
    :param scala_max: maximum value of the desired scale

    :return: translated value
    """
    return (scala_max - scala_min) * original_value + scala_min

# Translate dataframe points according to the range of the cells
for idx, df_screen in enumerate(df_tot):
    for num, df_sub in enumerate(df_screen):
        n_biods = len(df_sub.columns) // 2 - 1
        for i in range(n_biods):
            df_sub[f'biod{i}_x'] = translate(df_sub[f'biod{i}_x'], x_RANGE_CELL_px[num][0], x_RANGE_CELL_px[num][1])
            df_sub[f'biod{i}_y'] = translate(df_sub[f'biod{i}_y'], y_RANGE_CELL_px[0], y_RANGE_CELL_px[1])


# Initialize objects
# 1. Initialize windows
# setting origin: bottom left, coordinates: x values positive on the right, y values positive going up
window_1 = visual.Window([1920, 1080], color=(255, 255, 255), viewPos=(-PIXELS_MONITOR[0] / 2, - PIXELS_MONITOR[1] / 2),
                         fullscr=True, units='pix', screen=1, monitor=myMon)
window_2 = visual.Window([1920, 1080], color=(255, 255, 255), viewPos=(-PIXELS_MONITOR[0] / 2, - PIXELS_MONITOR[1] / 2),
                         fullscr=True, units='pix', screen=2, monitor=myMon)

# 2. Initialize red lines for setup grid
windows_list = [window_1, window_2]
lines = []

for window in windows_list:
    for i in range(NUMBER_OF_LINES):
        rect = visual.Rect(window, width=width_lines_px, height=length_lines_px,
                           pos=(coordinates_lines_px[i][0], coordinates_lines_px[i][1]), fillColor='red')
        lines.append(rect)

# 3. Initialize biods
biods_tot = []
size_dots = tools.monitorunittools.cm2pix(data_dict['stimuli_parameters']['size_dots_cm'], myMon)

for idx, window in enumerate(windows_list):
    screen_biods_list = []
    for n in range(NUM_CHAMBERS):
        biods_sublist = []
        n_biods = len(df_tot[idx][n].columns) // 2 - 1
        for i in range(n_biods):
            stimulus = visual.Circle(window, size=size_dots, pos=[0, 0], fillColor='black')
            biods_sublist.append(stimulus)
        screen_biods_list.append(biods_sublist)
    biods_tot.append(screen_biods_list)

# 4. Initialize video
if data_dict['experiment_metadata']['experimental_group'] == 'exp-realfish-empty':
    zebra_video = r"\\cimec-storage5\acn_lab\shared_acn\Matilde\zebrafish\biodi_experiment\stimuli\zebra_6wpf_group.mp4"
    emptytank_video = r"\\cimec-storage5\acn_lab\shared_acn\Matilde\zebrafish\biodi_experiment\stimuli\empty_tank.mp4"

    video1_win1 = visual.MovieStim(window_1, filename=zebra_video, pos=(18.5, 3.9), anchor='center', size=(6, 6), units='cm')
    video2_win1 = visual.MovieStim(window_1, filename=zebra_video, pos=(25, 3.9), anchor='center', size=(6, 6), units='cm')
    video3_win1 = visual.MovieStim(window_1, filename=zebra_video, pos=(31.5, 3.9), anchor='center', size=(6, 6), units='cm')

    video1_win2 = visual.MovieStim(window_2, filename=emptytank_video, pos=(18.5, 3.9), anchor='center', size=(6, 6), units='cm')
    video2_win2 = visual.MovieStim(window_2, filename=emptytank_video, pos=(25, 3.9), anchor='center', size=(6, 6), units='cm')
    video3_win2 = visual.MovieStim(window_2, filename=emptytank_video, pos=(31.5, 3.9), anchor='center', size=(6, 6), units='cm')

# Workflow presentation
# 1. Draw grids for setup alignment
for rect in lines:
    rect.draw()

window_1.flip()
window_2.flip()
event.waitKeys()

# 2. Initial Habituation Pause
pause_clock = core.Clock()

while pause_clock.getTime() < data_dict['stimuli_parameters']['pause_duration']:
    window_1.flip()
    window_2.flip()
    remaining_time = data_dict['stimuli_parameters']['pause_duration'] - pause_clock.getTime()
    if (data_dict['stimuli_parameters']['pause_duration'] - pause_clock.getTime()) < 360:
        print(f'We are starting in {remaining_time/60} minutes')
    if 'escape' in event.getKeys():
        break

# 3. Biods Presentation
t = df_screen1['t'] / 30   # 30 frame per second

expClock = core.Clock()
data_dict['experiment_metadata']['stimuli_start_time'] = data.getDateStr(format="%H%M%S")
while expClock.getTime() < data_dict['stimuli_parameters']['stimuli_duration']:
    elapsed_t = expClock.getTime()

    if data_dict['experiment_metadata']['experimental_group'] == 'exp-realfish-empty':
        video1_win1.draw()
        video2_win1.draw()
        video3_win1.draw()
        video1_win2.draw()
        video2_win2.draw()
        video3_win2.draw()
    else:
        for screen, df_screen in enumerate(df_tot):
            for num, df_sub in enumerate(df_screen):
                n_biods = len(df_sub.columns) // 2 - 1
                for i in range(n_biods):
                    values = df_sub[f'biod{i}_x'], df_sub[f'biod{i}_y']
                    interpolated_function = scipy.interpolate.interp1d(t, values)
                    current_value = interpolated_function(elapsed_t)
                    biods_tot[screen][num][i].pos = current_value

        for screen in biods_tot:
            for sublist in screen:
                for biod in sublist:
                    biod.draw()

    window_1.flip()
    window_2.flip()

    if 'escape' in event.getKeys():
        break

print(f'Stimuli are ended')

data_dict['experiment_metadata']['stimuli_end_time'] = data.getDateStr(format="%H%M%S")

# Create a DataFrame using the dictionary
metadata_df = pd.concat([pd.DataFrame(data_dict['experiment_metadata'], index=[0]),
                         pd.DataFrame(data_dict['stimuli_parameters'], index=[0])], axis=1)

if data_dict['experiment_metadata']['experiment_name'] != "":
    metadata_df.to_csv(new_path / filename, index=False)

window_1.flip()
window_2.flip()
event.waitKeys()

window_1.close()
window_2.close()
