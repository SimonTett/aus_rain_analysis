# make all the plots
import matplotlib.pyplot as plt
import pathlib
files = list(pathlib.Path('plots').glob('plot_*.py'))
for file in files:
    expect_plot = pathlib.Path('figures') / (file.stem.replace('plot_','') + '.png')
    print(expect_plot)
    if expect_plot.exists():
        print(f'Skipping {file} as {expect_plot} exists')
        continue
    print(f'Running {file}')
    runfile(str(file))
    plt.close('all') # close all the figures

