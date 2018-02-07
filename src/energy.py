from numpy import *
from numpy import min as nmin
import config

try:
    from pycuda import compiler, gpuarray
    import pycuda.autoinit
except ImportError:
    pass

indexDiv = []
indexMod = []
sysEn = None


# *** CPU CODE ***


def init(pc):
    global indexDiv, indexMod, sysEn
    indexDiv = [j // pc for j in range(pc ** 2)]
    indexMod = [j % pc for j in range(pc ** 2)]
    if config.device == "cpu":
        sysEn = cpuSysEn
    elif config.device == "gpu":
        sysEn = gpuSysEn
    else:
        raise ValueError("Invalid value of 'device'")


def module_matrix(xpos, ypos):
    size = xpos.size
    xr1 = xpos[indexDiv]
    xr2 = xpos[indexMod]
    yr1 = ypos[indexDiv]
    yr2 = ypos[indexMod]
    return triu(linalg.norm(vstack([xr1 - xr2, yr1 - yr2]),
                            axis = 0).reshape(size, size))


def angle_matrix(xpos, ypos, vec):
    size = xpos.size

    xr1 = xpos[indexDiv]
    xr2 = xpos[indexMod]
    yr1 = ypos[indexDiv]
    yr2 = ypos[indexMod]
    scalar = (xr1 - xr2) * vec[0] + (yr1 - yr2) * vec[1]
    norm = linalg.norm(vstack([xr1 - xr2, yr1 - yr2]), axis = 0)
    norm[norm == 0.0] = inf

    return triu((scalar / (1.0 * norm)).reshape(size, size))


def cpuSysEn(positions):
    angles = angle_matrix(positions[0], positions[1], config.field_direction)

    modules = module_matrix(positions[0], positions[1])
    modules[modules == 0.0] = inf
    if modules[modules < config.diskRadius * 2].size > 0:
        return inf

    esum = 0.0
    if config.potential == "dipole":
        esum = (config.diskRadius ** 3) * sum((1.0 - 3 * angles ** 2) / (modules ** 3))
    elif config.potential == "quadropole":
        esum = (config.diskRadius ** 5) * sum((35.0 * angles ** 4 - 30.0 * angles ** 2 + 3.0) / (modules ** 5))
        #print "esum1", esum
    else: raise ValueError("Invalid value of 'potential'")
    return esum


def closest_image(x, y, x1, y1):
    L = 2.0

    r = [(x - x1) ** 2 + (y - y1) ** 2,              # ( 0  0 )
         (x - x1) ** 2 + (y - (y1 + L)) ** 2,        # ( 0  L )
         (x - (x1 + L)) ** 2 + (y - (y1 + L)) ** 2,  # ( L  L )
         (x - (x1 + L)) ** 2 + (y - y1) ** 2,        # ( L  0 )
         (x - (x1 + L)) ** 2 + (y - (y1 - L)) ** 2,  # ( L -L )
         (x - x1) ** 2 + (y - (y1 - L)) ** 2,        # ( 0 -L )
         (x - (x1 - L)) ** 2 + (y - (y1 - L)) ** 2,  # (-L -L )
         (x - (x1 - L)) ** 2 + (y - y1) ** 2,        # (-L  0 )
         (x - (x1 - L)) ** 2 + (y - (y1 + L)) ** 2]  # (-L  L )

    return sqrt(nmin(r, axis=0))


def mu(positions, idx):
    vec = [[config.field_direction[0]], [config.field_direction[1]]]
    pc = positions[:,idx].reshape(2,1)

    modules = closest_image(pc[0][0], pc[1][0], positions[0], positions[1])

    # modules = linalg.norm(positions - pc, axis = 0)
    # modules2 = []
    # for j in range(positions[0].size):
    #    modules2.append(closest_image(pc[0][0], pc[1][0], positions[0][j], positions[1][j]))
    # modules2 = array(modules2)

    modules[modules == 0.0] = inf
    if modules[modules < config.diskRadius * 2].size > 0: return inf
    angles = sum((positions - pc) * vec, axis=0) / modules

    esum = 0.0
    if config.potential == "dipole":
        esum = (config.diskRadius ** 3) * sum((1.0 - 3 * angles ** 2) / (modules ** 3))
    elif config.potential == "quadropole":
        esum = (config.diskRadius ** 5) * sum((35.0 * angles ** 4 - 30.0 * angles ** 2 + 3.0) / (modules ** 5))
    else: print("Error")
    return esum


# *** GPU CODE ***

g_norm, g_angle, g_demo = None, None, None
kernel_code = """
#include <cmath>
#define PI 3.14159265358979323846

__global__ void pairsNorm(float *ax, float *ay, float *c) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int offset = tx + bx * blockDim.x;
    if (bx < tx)
        //c[tx + bx * %(matrixSize)s] = sqrt(pow(ax[bx] - ax[tx], 2) +
        //									pow(ay[bx] - ay[tx], 2));
        c[offset] = sqrt(pow(ax[bx] - ax[tx], 2) +
                                            pow(ay[bx] - ay[tx], 2));
}

// debug
__global__ void pairsDemo(float *ax, float *ay, float *c) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int offset = tx + bx * blockDim.x;
    if (bx < tx)
        c[offset] = 10 * bx + tx;
}

__global__ void pairsAngle(float *ax, float *ay, float *c) {
    float vx = 0.0, vy = 1.0;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    if (bx < tx) {
        float differenceVecX = ax[bx] - ax[tx];
        float differenceVecY = ay[bx] - ay[tx];
        float scalarMul = vx * differenceVecX + vy * differenceVecY;
        float differenceNorm = sqrt(pow(differenceVecX, 2) + pow(differenceVecY, 2));
        c[tx + bx * %(matrixSize)s] = scalarMul / (1.0 * differenceNorm);
        //c[tx + bx * %(matrixSize)s] = 180 / PI * acos(scalarMul / (1.0 * differenceNorm));
    }
}
"""


def cudaInit(kernel_code):
    global cuda_norm, cuda_angle, cuda_demo
    kernel_code = kernel_code % {
    'matrixSize': matrixSize
    }
    mod = compiler.SourceModule(kernel_code)
    g_norm = mod.get_function("pairsNorm")
    g_angle = mod.get_function("pairsAngle")
    g_demo = mod.get_function("pairsDemo")


def gpuSysEn(positions):
    xpos_gpu = gpuarray.to_gpu(positions[0].astype(float32))
    ypos_gpu = gpuarray.to_gpu(positions[1].astype(float32))

    c_gpu = gpuarray.empty((matrixSize * matrixSize), float32)
    d_gpu = gpuarray.empty((matrixSize * matrixSize), float32)

    g_norm(xpos_gpu, ypos_gpu, c_gpu, block = (matrixSize, 1, 1), grid = (matrixSize, 1))
    g_angle(xpos_gpu, ypos_gpu, d_gpu, block = (matrixSize, 1, 1), grid = (matrixSize, 1))

    modulesMatrix = c_gpu.get().reshape(matrixSize, matrixSize)
    anglesMatrix = d_gpu.get().reshape(matrixSize, matrixSize)

    modulesMatrix[modulesMatrix == 0.0] = inf

    result = sum((35.0 * anglesMatrix ** 4 - 30.0 * anglesMatrix ** 2 + 3.0) / (modulesMatrix ** 5))
    return result

matrixSize = config.pointsPerLine ** 2