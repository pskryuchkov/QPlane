
# physical parameters
temperature = 0.005
#temperature = 8.65 * 10 ** 6 # ~ 0.2 * 10 ** 6
field_direction = [0, 1]
diskRadius = 0.020 # ~0.02

# modelling parameters
maxDeltaPos = diskRadius # ~ 0.02

steps = 150100
passSteps = 100
statPeriod = 1000
"""
steps = 500
passSteps = 100
statPeriod = 100
"""
# accuracy of printing
float_accuracy = 9

# box parameters
pic_limits = 1.4
box_size = hor_lim = ver_lim = 2.0
bounds = "tight" # fixed / tight

# on fly generation parameters
n_vertical = 25
n_horizontal = 40

potential = "quadropole" # quadropole / dipole

# flags
save_log = 1
save_screenshot = 1
measure_correlations = 0
measure_energy = 1
measure_deviations = 1
save_positions = 1

device = "cpu"

init_positions = "lattice" # probe / last / lattice

# input / output files
in_probe = "../data/probe/probe_positions.txt"
last_positions   = "../data/last/last_positions.txt"
out_correlations = "corr_func.txt"
out_energy       = "energy.txt"
out_screen       = "screenshot.png"
log_file         = "log.txt"
out_dir          = "out"

# deprecated
pointsPerLine = 3
pointsCount = 60
chainN = 10
chainL = 50
boxSize = 2.0

# profiler calling
# python -m cProfile -s tottime Dynamic.py > log.txt

tp = 0.1
dx = diskRadius

blocks = 500
n_iter = 500


