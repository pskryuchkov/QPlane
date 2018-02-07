#!/usr/bin/python

from parser import config as uconfig
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from numpy import *
import warnings
import getopt
import config
import energy
import time
import sys
import os


global_energy = None
start_time = None
_print = print


def save_point(filename, x, y, xerr=0.0, yerr=0.0):
    with open(filename, 'a') as logfile:
        logfile.write("{0} {1} {2} {3}\n".format(x, y, xerr, yerr))


def pair_correlator(positions):
    xpos = positions[0]
    ypos = positions[1]
    size = xpos.size
    xr1 = xpos[indexDiv]
    xr2 = xpos[indexMod]
    yr1 = ypos[indexDiv]
    yr2 = ypos[indexMod]
    return vstack([xr1 - xr2, yr1 - yr2]).T


class ParticleSystem:
    def __init__(self, init_coords, temperature, energy_func):

        self.n_particles = init_coords.shape[1]
        self.prev_en, self.curr_en = 0, 0
        self.temperature = temperature

        self._success_cnt, self._fail_cnt = 0, 0
        self._positions = init_coords
        self._scene = Scene(self)

    def get_coords(self):
        return self._positions[0], self._positions[1]

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
        new_positions = copy(self._positions)
        index = random.randint(0, self.n_particles)

        new_positions[0, index] += random.uniform(low=-max_dx, high=max_dx)

        new_positions[1, index] += random.uniform(low=-max_dx, high=max_dx)

        return new_positions, index

    def metropolis_step(self, t, dx):
        global global_energy

        if global_energy is None:
            global_energy = energy.sysEn(self._positions)

        cur_energy = global_energy

        while True:
            new_positions, idx = self.move_one(dx)

            self.keep_bounds(new_positions, config.hor_lim, config.ver_lim, idx)

            old_mu = energy.mu(self._positions, idx)
            new_mu = energy.mu(new_positions, idx)

            if new_mu == inf:
                continue

            new_energy = cur_energy - old_mu + new_mu

            delta = new_energy - cur_energy

            if self._fail_cnt > 0 and self._fail_cnt % 500 == 0:
                warnings.warn("Warning: too many unsuccessful moves", UserWarning)

            if delta < 0:
                self._positions = copy(new_positions)
                self._success_cnt += 1
                global_energy = new_energy

                return new_energy
            else:
                r = random.uniform()
                if r < exp(-delta / t):
                    self._positions = copy(new_positions)
                    self._success_cnt += 1
                    global_energy = new_energy

                    return new_energy
                else:
                    self._fail_cnt += 1

    def print_info(self, iter):

        acception_rate = float32(100.0 * self._success_cnt) / \
                         (self._success_cnt + self._fail_cnt)

        energy_per_particle = self.curr_en / self.n_particles

        delta_energy = (self.prev_en - energy_per_particle) / energy_per_particle

        mins, secs = int((time.time() - start_time) // 60), \
                     int((time.time() - start_time) % 60)

        print("{0} {1:02d}:{2:02d} E={3:.3f} dE={4:.2e}% ({5:.2f}%)".
            format(iter, mins, secs, energy_per_particle, delta_energy, acception_rate))

        self.prev_en = energy_per_particle

    def evolution(self):

        coord_evolution = []

        for b in range(uconfig.blocks):

            self._success_cnt, self._fail_cnt = 0, 0

            for i in range(uconfig.n_iter):
                self.curr_en = self.metropolis_step(uconfig.tp, uconfig.dx)

            self.print_info(b)
            self._scene.shot()

            coord_evolution.append(self._positions)

        if config.measure_deviations:
            pass

        if config.measure_correlations:
            pass

        if config.save_coordinates:
            pass

    def freezing(self, start_pow=2, end_pow=-3, dt_steps=10,
                 dx_factor=0.7, balance_wait_blocks=30, stop_rate=0.07):

        coord_evolution = []

        t_val = logspace(start_pow, end_pow, dt_steps)

        idx = 0
        curr_t, curr_dx = t_val[idx], uconfig.dx

        for b in range(dt_steps * balance_wait_blocks):
            self._success_cnt, self._fail_cnt = 0, 0

            if b % balance_wait_blocks == 0:
                if b > 0:
                    idx += 1
                    curr_t, curr_dx = t_val[idx], curr_dx * dx_factor

                print("T={} Dx={}".format(curr_t, curr_dx))

            e_eval = []
            for i in range(uconfig.n_iter):
                e = self.metropolis_step(curr_t, curr_dx)
                e_eval.append(e)

            self.curr_en = mean(e_eval)

            self.print_info(b)
            self._scene.shot()

            coord_evolution.append(self._positions)

            acception_rate = float32(100.0 * self._success_cnt) / \
                             (self._success_cnt + self._fail_cnt)

            if acception_rate < stop_rate:
                break


class Scene():
    def __init__(self, physical_system):
        self.ps = physical_system
        self.fig, self.ax = None, None

    def create_window(self):
        mpl.rcParams['toolbar'] = 'None'
        plt.xlim(-config.pic_limits, config.pic_limits)
        plt.ylim(-config.pic_limits, config.pic_limits)

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.set_window_title('Monte-Carlo Simulation Progress')

    def shot(self):
        if self.fig is None or self.ax is None:
            self.create_window()

        xc, yc = self.ps.get_coords()
        self.ax.plot(xc, yc, "o")
        plt.pause(.01)
        plt.draw()
        plt.cla()

    def __exit__(self):
        if config.save_screenshot:
            self.fig.savefig('shot.png', dpi=300)


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def warning_format(warn_msg, *a):
    return str(warn_msg) + '\n'


def reset_file(fn):
    with open(fn, "w"):
        pass


def print(*args):
    _print(*args)

    if config.save_log:
        with open(config.log_file, "a") as f:
            _print(*args, file=f)


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
            print("out directory: {0}".format(config.out_dir))
        if opt in ('-p', '--positions'):
            config.init_positions = arg
        if opt in ('-m', '--max_delta'):
            config.maxDeltaPos = float(arg)


def set_paths():
    out_root = Path(config.out_dir)
    config.log_file = out_root / config.log_file
    config.out_screen = out_root / config.out_screen
    config.out_energy = out_root / config.out_energy
    config.out_correlations = out_root / config.out_correlations


def rect_lattice():
    lx, ly = [], []
    tight_factor = 0.96

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
        raise ValueError("Invalid value of 'bounds'")

if __name__ == "__main__":

    scriptDir = os.path.dirname(os.path.realpath(__file__))
    warnings.formatwarning = warning_format

    parse_args()
    set_paths()
    reset_file(config.log_file)

    print("potential: {}".format(config.potential))
    # print([x for x in list(config.__dict__.keys()) if not x.startswith("_")])

    if not os.path.isdir(config.out_dir):
        os.makedirs(config.out_dir)

    positions = load_coordinates()

    # log("potential:", config.potential)
    # log("n_particles:", len(positions[0]))
    # log("temperature: {0:.2e}".format(config.temperature * config.diskRadius ** (-5)))

    # log("hx={0}, hy={1}".format(round(config.field_direction[0], 3),
    #                            round(config.field_direction[1], 3)))
    set_bounds()

    # log("delta_pos:", config.maxDeltaPos)
    # log("disk_radius:", config.diskRadius)

    positions = positions.astype(double)

    print("\nStarting simulation...")

    liquid = ParticleSystem(positions, config.temperature)

    energy.init(liquid.n_particles)

    start_time = time.time()

    if uconfig.scene == "'freezing'":
        liquid.freezing()
    elif uconfig.scene == "'evolution'":
        liquid.evolution()
    else:
        raise ValueError("Invalid value of 'scene': {}".format(uconfig.scene))

    end = time.time()

    # log("te:\t", float_to_string(end - start_time, 4))
    # log("dv:\t", float_to_string(devs, 6))

    # if config.save_screenshot:
    #    plt.plot(liquid.positions[0], liquid.positions[1], "o",
    #             markeredgewidth=0.0, alpha=0.8, markersize=5.3)
    #
    #    plt.savefig(config.out_screen, dpi=200)

    # if config.save_positions:
    #    savetxt(config.last_positions, liquid.positions, delimiter=',')
    #    savetxt(config.out_dir + "/" + "last_positions.txt",
    #            liquid.positions, delimiter=',')

    # if config.measure_correlations:
    #    savetxt(config.out_correlations, stat, delimiter=',')

    # if config.measure_energy:
    #    savetxt(config.out_energy, energy_list, delimiter=',')

    # save_log()
