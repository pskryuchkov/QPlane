#!/usr/bin/python

from numpy import *
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
import config
import energy
import warnings
import getopt

start_time = None
global_energy = None
log_string = ""


def get_draw_context():
    mpl.rcParams['toolbar'] = 'None'
    fig = plt.figure(figsize=(12, 8))
    axes = plt.gca()
    ax = fig.add_subplot(111)
    return fig, ax


def save_point(filename, x, y, xerr=0.0, yerr=0.0):
    with open(filename, 'a') as logfile:
        logfile.write("{0} {1} {2} {3}\n".format(x, y, xerr, yerr))


def clear_file(filename):
    open(filename, 'w').close()


def dot_product(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def vec_length(v):
    return sqrt(dot_product(v, v))


def angle_bv(v1, v2):
    if vec_length(v1) * vec_length(v2) == 0:
        return 0

    cos_val = dot_product(v1, v2) / (vec_length(v1) * vec_length(v2))

    if -1.0 <= cos_val <= 1.0:
        return arccos(dot_product(v1, v2) / (vec_length(v1) * vec_length(v2)))

    else:
        return pi if cos_val > 1.0 else -pi


def cos_func(v1, v2):
    if vec_length(v1) * vec_length(v2) == 0:
        return 0
    return dot_product(v1, v2) / (vec_length(v1) * vec_length(v2))

##################################################################################


def correlator(positions):
    xpos = positions[0]
    ypos = positions[1]
    size = xpos.size
    xr1 = xpos[indexDiv]
    xr2 = xpos[indexMod]
    yr1 = ypos[indexDiv]
    yr2 = ypos[indexMod]
    return vstack([xr1 - xr2, yr1 - yr2]).T


# FIXME : remove this
def float_to_string(n, ac=config.float_accuracy):
    return ("{0:." + str(ac) + "f}").format(n)


class ParticleSystem:
    prev_energy = 0

    if __name__ == "__main__":
        fig, ax = get_draw_context()

    def __init__(self, init_coords, temperature, energy_func):
        self.firstFrame = True
        self.initPos = copy(init_coords)
        self.positions = init_coords
        self.n_particles = size(self.positions[0])
        self.energyFunc = energy_func
        self.temperature = temperature
        self.success_cnt = 0
        self.fail_cnt = 0
        self.initSingle = []
        self.curr_en = 0

    # FIXME, CHECKME
    def keep_bounds(self, prob_coords, xlim, ylim, j):

        if prob_coords[0][j] > xlim / 2.0:
            prob_coords[0][j] -= xlim

        if prob_coords[0][j] < -xlim / 2.0:
            prob_coords[0][j] += xlim

        if prob_coords[1][j] > ylim / 2.0:
            prob_coords[1][j] -= ylim

        if prob_coords[1][j] < -ylim / 2.0:
            prob_coords[1][j] += ylim

    def move_one(self, max_dx):
        new_positions = copy(self.positions)
        index = random.randint(0, self.n_particles)

        new_positions[0, index] += random.uniform(low=-max_dx, high=max_dx)

        new_positions[1, index] += random.uniform(low=-max_dx, high=max_dx)

        return new_positions, index

    def metropolis_step(self, t, dx):

        global global_energy

        if global_energy is None:
            global_energy = energy.sysEn(self.positions)

        cur_energy = global_energy

        while True:
            new_positions, idx = self.move_one(dx)

            self.keep_bounds(new_positions, config.hor_lim, config.ver_lim, idx)

            old_mu = energy.mu(self.positions, idx)
            new_mu = energy.mu(new_positions, idx)

            if new_mu == inf:
                continue

            new_energy = cur_energy - old_mu + new_mu

            delta = new_energy - cur_energy

            if self.fail_cnt > 0 and self.fail_cnt % 500 == 0:
                warnings.warn("Warning: too many unsuccessful moves", UserWarning)

            if delta < 0:
                self.positions = copy(new_positions)
                self.success_cnt += 1
                global_energy = new_energy

                return new_energy
            else:
                r = random.uniform()
                if r < exp(-delta / t):
                    self.positions = copy(new_positions)
                    self.success_cnt += 1
                    global_energy = new_energy

                    return new_energy
                else:
                    self.fail_cnt += 1

    def draw_crystal(self, frame_size):
        if self.firstFrame:
            fig = plt.gcf()
            fig.canvas.set_window_title('Monte-Carlo Simulation Progress')

            plt.ion()
            plt.xlim(-frame_size, frame_size)
            plt.ylim(-frame_size, frame_size)

            plt.show()
            self.firstFrame = False
            plt.plot([-1.0, -1.0 + config.diskRadius * 2], [-1.0, -1.0])

        descriptors_list = []

        for j in range(self.n_particles):
            if j in self.initSingle:
                descriptors_list.append(ParticleSystem.ax.scatter(
                                    [self.positions[0, j]],
                                    [self.positions[1, j]],
                                    color="b", edgecolors='w', s=30, alpha=0.7))
            else:
                descriptors_list.append(ParticleSystem.ax.scatter(
                                    [self.positions[0, j]],
                                    [self.positions[1, j]],
                                    color="r", edgecolors='w', s=30, alpha=0.7))

        ParticleSystem.fig.canvas.draw()

        for j in range(len(descriptors_list)):
            descriptors_list[j].remove()

    # FIXME: remove this
    def show_progress(self):
        self.draw_crystal(frame_size=config.pic_limits)

    def write_log(self, iter):

        acception_rate = float32(100.0 * self.success_cnt) / \
                         (self.success_cnt + self.fail_cnt)

        energy_per_particle = self.curr_en / self.n_particles

        delta_energy = (self.prev_energy - energy_per_particle) / energy_per_particle

        mins, secs = int((time.time() - start_time) // 60), \
                     int((time.time() - start_time) % 60)

        log("{0} {1:02d}:{2:02d} E={3:.3f} dE={4:.2e}% ({5:.2f}%)".
            format(iter, mins, secs, energy_per_particle, delta_energy, acception_rate))

        self.prev_energy = energy_per_particle

    def evolution(self):

        coord_evolution = []

        for b in range(config.blocks):

            self.success_cnt, self.fail_cnt = 0, 0

            for i in range(config.n_iter):
                self.curr_en = self.metropolis_step(config.temperature,
                                                   config.maxDeltaPos)
            self.show_progress()
            self.write_log(b)

            coord_evolution.append(self.positions)

        if config.measure_deviations:
            pass

        if config.measure_correlations:
            pass

        if config.save_coordinates:
            pass

        if config.save_screenshot:
            pass

    def freezing(self, tp_factor=0.7, dx_factor=0.7, balance_wait_blocks=30):

        curr_t, curr_dx = config.tp, config.dx
        coord_evolution = []

        for b in range(config.blocks):
            self.success_cnt, self.fail_cnt = 0, 0

            if b % balance_wait_blocks == 0:
                if b > 0:
                    curr_t, curr_dx = curr_t * tp_factor, curr_dx * dx_factor

                log("T={} Dx={}".format(curr_t, curr_dx))

            for i in range(config.n_iter):
                self.curr_en = self.metropolis_step(curr_t, curr_dx)

            self.show_progress()
            self.write_log(b)

            coord_evolution.append(self.positions)

        if config.save_screenshot:
            pass


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def refresh_reports(filename):
    clear_file(filename) if os.path.isfile(filename) else touch(filename)

##################################################################################


def warning_format(warn_msg, *a):
    return str(warn_msg) + '\n'


def log(*args):
    global log_string
    for j, arg in enumerate(args):
        if type(arg) != 'str':
            print(str(arg)),
            if config.save_log:
                log_string += str(arg) + " "
        else:
            print(arg),
            if config.save_log:
                log_string += arg + " "
    print
    if config.save_log:
        log_string += "\n"


def save_log():
    global log_string
    if log_string != "":
        with open(config.log_file, "w") as log_file:
            log_file.write(log_string)


def parse_args():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 't:r:d:p:m:',
                                   ['temperature=', 'radius=', 'directory=', 'positions='])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-t', '--temperature'):
            config.temperature = float(arg)
        if opt in ('-r', '--radius'):
            config.diskRadius = float(arg)
        if opt in ('-d', '--directory'):
            config.out_dir = arg
            log("out directory: {0}".format(config.out_dir))
        if opt in ('-p', '--positions'):
            config.init_positions = arg
        if opt in ('-m', '--max_delta'):
            config.maxDeltaPos = float(arg)


def set_paths():
    config.out_correlations = config.out_dir + "/" + config.out_correlations
    config.out_screen = config.out_dir + "/" + config.out_screen
    config.log_file = config.out_dir + "/" + config.log_file
    config.out_energy = config.out_dir + "/" + config.out_energy


def rect_lattice():
    lx, ly = [], []
    tight_factor = 0.95

    x_order, y_order = config.n_horizontal, config.n_vertical
    side = config.box_size

    for v in range(x_order):
        for w in range(y_order):
            lx.append(v * side / (x_order - 1))
            ly.append(w * side / (y_order - 1))

    lx = asarray(lx) - 1.0
    ly = asarray(ly) - 1.0

    return vstack([lx, ly]) * tight_factor


def load_coordinates():
    init_coords = None
    log("mode: '{}'".format(config.init_positions))
    if config.init_positions == "lattice":
        init_coords = rect_lattice()
    elif config.init_positions == "probe":
        init_coords = loadtxt(config.in_probe, delimiter=',')
    elif config.init_positions == "last":
        init_coords = loadtxt(config.last_positions, delimiter=',')
    else:
        pass
    return init_coords


def set_bounds():
    if config.bounds == "fixed":
        pass
    elif config.bounds == "tight":
        config.hor_lim = max(positions[0]) * 2.01
        config.ver_lim = max(positions[1]) * 2.01
    else:
        print("Error")

if __name__ == "__main__":

    scriptDir = os.path.dirname(os.path.realpath(__file__))
    warnings.formatwarning = warning_format
    parse_args()
    set_paths()
    if not os.path.isdir(config.out_dir):
        os.makedirs(config.out_dir)

    positions = load_coordinates()

    log("potential:", config.potential)
    log("n_particles:", len(positions[0]))
    log("temperature: {0:.2e}".format(config.temperature * config.diskRadius ** (-5)))

    log("hx={0}, hy={1}".format(round(config.field_direction[0], 3),
                                round(config.field_direction[1], 3)))
    set_bounds()

    log("delta_pos:", config.maxDeltaPos)
    log("disk_radius:", config.diskRadius)

    positions = positions.astype(double)

    log("\nStarting simulation...")

    liquid = ParticleSystem(positions, config.temperature, None)

    energy.init(liquid.n_particles)

    start_time = time.time()

    #liquid.evolution()
    liquid.freezing()

    end = time.time()

    #log("te:\t", float_to_string(end - start_time, 4))
    #log("dv:\t", float_to_string(devs, 6))

    #if config.save_screenshot:
    #    plt.plot(liquid.positions[0], liquid.positions[1], "o",
    #             markeredgewidth=0.0, alpha=0.8, markersize=5.3)
    #
    #    plt.savefig(config.out_screen, dpi=200)

    #if config.save_positions:
    #    savetxt(config.last_positions, liquid.positions, delimiter=',')
    #    savetxt(config.out_dir + "/" + "last_positions.txt",
    #            liquid.positions, delimiter=',')

    #if config.measure_correlations:
    #    savetxt(config.out_correlations, stat, delimiter=',')

    #if config.measure_energy:
    #    savetxt(config.out_energy, energy_list, delimiter=',')

    save_log()
