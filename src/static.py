from numpy import *
import matplotlib.pyplot as plt
import config
import os

def modulesMatrix(xpos, ypos):
    size = xpos.size
    indexDiv = [j / size for j in range(size ** 2)]
    indexMod = [j % size for j in range(size ** 2)]
    xr1 = xpos[indexDiv]
    xr2 = xpos[indexMod]
    yr1 = ypos[indexDiv]
    yr2 = ypos[indexMod]
    return triu(linalg.norm(vstack([xr1 - xr2, yr1 - yr2]), axis = 0).reshape(size, size))

def anglesMatrix(xpos, ypos, vec):
    size = xpos.size
    indexDiv = [j / size for j in range(size ** 2)]
    indexMod = [j % size for j in range(size ** 2)]
    xr1 = xpos[indexDiv]
    xr2 = xpos[indexMod]
    yr1 = ypos[indexDiv]
    yr2 = ypos[indexMod]
    scalar = (xr1 - xr2) * vec[0] + (yr1 - yr2) * vec[1]
    norm = linalg.norm(vstack([xr1 - xr2, yr1 - yr2]), axis = 0)
    norm[norm == 0.0] = inf
    return triu((scalar / (1.0 * norm)).reshape(size, size))

def systemEnergy(positions):
    #c49 = 0.65605902899
    #s49 = 0.70710678118
    mm = modulesMatrix(positions[0], positions[1])
    am = anglesMatrix(positions[0], positions[1], [0., 1.])
    mm[mm == 0.0] = inf
    if mm[mm < config.diskRadius * 2].size > 0: return inf
    # quadrupoles
    #msum = sum((1.0 - 3.0 * am ** 2) / (mm ** 3))
    msum = sum((35.0 * am ** 4 - 30.0 * am ** 2 + 3.0) / (mm ** 5))
    return msum

def particlesEnergy(positions, particleIndexes):
    mm = modulesMatrix(positions[0], positions[1])
    am = anglesMatrix(positions[0], positions[1], [0.0, 1.0])
    mm[mm == 0.0] = inf
    if mm[mm < diskRadius * 2].size > 0: return inf

    ####  patch 3 here  ####
    eMatrix = (35.0 * am ** 4 - 30.0 * am ** 2 + 3.0) / (mm ** 5)
    teMatrix = eMatrix.T
    energies = {}

    for idx in particleIndexes:
        energy = 0.0

        for k in range(idx - 1):
            if isfinite(teMatrix[idx][k]):
                energy += teMatrix[idx][k]

        for g in range(idx + 1, len(eMatrix[idx])):
            if isfinite(eMatrix[idx][g]):
                energy += eMatrix[idx][g]

        energies[idx] = energy

    return energies

def drawSquare(side, c = 'b'):
    plt.plot(side * array([-1.0, -1.0, 1.0, 1.0, -1.0]),
        side * array([-1.0, 1.0, 1.0, -1.0, -1.0]), c, lw=0.5)

def diagLattice():
    x, y = [], []
    chains = 5
    perChain = 20
    side = 2.0
    sX, sY = 0.0, 0.0
    shift = 0.5

    for j in range(chains):
        for i in range(perChain):
            xp = sX + cos(30.*pi/180) * i * side / perChain + shift * j
            yp = sY + cos(30.*pi/180 ) * i * side / perChain
            if abs(xp) < side and abs(yp) < side:
                x.append(xp)
                y.append(yp)
            """
            if j > 0:
                xp = sX + i * side / perChain
                yp = sY + i * side / perChain + shift * j
                if abs(xp) < side and abs(yp) < side:
                    x.append(xp)
                    y.append(yp)
            """
    return x, y

def pairChainLattice():
    pointsVer = 7
    pointsHor = 4

    xChainShift = 0.06
    yChainShift = 0.06

    xCoordShift = -0.90
    yCoordShift = -0.88

    margin = 0.1
    side = 2.0

    x, y = [], []
    yStep, distBetChains =  (side - margin) / pointsVer, (side - margin) / pointsHor

    for m in range(pointsHor):
        #shiftFlag = m % 2
        shiftFlag = 0
        for j in range(pointsVer):
            shiftFlag1 = j % 2
            x.append(0.0 + m * distBetChains)
            y.append(j * yStep + shiftFlag * yChainShift)

        for k in range(pointsVer):
            x.append(0.0 + xChainShift + m * distBetChains)
            y.append(k * yStep + yChainShift - shiftFlag * yChainShift)

    return x, y

def roLattice():
    x, y = [], []
    order = 5
    shift = 0.3
    for k in range(order):
        for b in range(order + 1):
            x.append(shift * b)
            y.append(shift / 2 + shift * k)

            x.append(shift / 2 + shift * k)
            y.append(shift * b)
    return x, y

def cross():
    x, y = [], []
    shift = 0.3
    x.append(shift / 4)
    y.append(shift / 4)

    x.append(shift - shift / 4)
    y.append(shift / 4)

    x.append(shift / 4)
    y.append(shift - shift / 4)

    x.append(shift - shift / 4)
    y.append(shift - shift / 4)

    x.append(shift / 2)
    y.append(shift / 2)

    return x, y

def reLattice():
    x, y = [], []
    xOrder, yOrder = config.n_horizontal,config.n_vertical
    side = 2.0

    for v in range(xOrder):
        for w in range(yOrder):
            x.append((v) * side / (xOrder - 1))
            y.append((w) * side / (yOrder - 1))

    print 1 * side / xOrder, 1 * side / yOrder
    return x, y

def dcLattice():
    x, y = [], []
    xOrder, yOrder = 5, 11
    side = 2.0
    shift = 0.1
    for v in range(xOrder):
        for w in range(yOrder):
            x.append((v + 0.5) * side / xOrder)
            x.append((v + 0.5) * side / xOrder + shift)
            y.append((w + 0.5) * side / yOrder)
            y.append((w + 0.5) * side / yOrder)
    return x, y



def inColour(x, y):
    x = array(x)
    y = array(y)
    positions = vstack([x, y])
    chainLen = 8
    chainIdx = 0
    selected = positions.T[chainIdx * chainLen: chainIdx * chainLen + chainLen]
    unselected = delete(positions.T, arange(chainIdx * chainLen, chainIdx * chainLen + chainLen), 0) # 0 means first coordinate
    selected = selected.T
    unselected = unselected.T
    #print positions
    #print selected
    #print unselected
    e1 = systemEnergy(unselected)
    print e1
    plt.plot(selected[0], selected[1], "o", c = "red", markeredgecolor='none')
    plt.plot(unselected[0], unselected[1], "o", c = "blue", markeredgecolor='none')

def keepBound(positions, size):
    for j in range(len(positions[0])):
        if positions[0][j] > size / 2.0:
            positions[0][j] -= size
        if positions[0][j] < -size / 2.0:
            positions[0][j] += size

        if positions[1][j] > size / 2.0:
            positions[1][j] -= size
        if positions[1][j] < -size / 2.0:
            positions[1][j] += size

def chainDelta(positions):
    chainIdx = 1
    chainLen = 15
    boxSide = 2.0
    #print positions
    # -0.4 ... 0.4
    sc = 100
    w = 0.28
    #w = 0.8
    eList = []
    for p in range(sc + 1):
        dx = p * w / sc - w / 2
        print dx
        newpos = positions
        newpos[0, arange(chainIdx * chainLen, chainIdx * chainLen + chainLen)] += dx
        #keepBound(newpos, 2.0)
        eList.append(systemEnergy(newpos))

    print eList
    print w
    plt.plot(arange(len(eList)) * w / sc - w / 2, eList, 'x')

def show(positions):
    chainIdx = 1
    chainLen = 15
    #plt.xlim([-0.2, 2.2])
    #plt.ylim([-0.2, 2.2])
    #positions[1, arange(chainIdx * chainLen, chainIdx * chainLen + chainLen)] += -0.1
    plt.plot(positions[0], positions[1], 'o', markeredgecolor='none')
    plt.show()

def reLattice2():
    x, y = [], []
    xOrder, yOrder = config.chainN, config.chainL
    side = 2.0
    shift = 1.0
    for v in range(xOrder):
        if v % 2 == 0:
            for w in range(yOrder):
                xc, yc = (v) * side / (xOrder - 1), (w) * side / yOrder
                x.append(xc)
                y.append(yc)
        else:
            for w in range(yOrder):
                xc, yc = (v) * side / (xOrder - 1), (w) * side / yOrder + (tan(0.68) - 1) * side / xOrder
                #if yc > side: yc = yc - side
                x.append(xc)
                y.append(yc)
    print 1 * side / yOrder, side / xOrder
    return x, y

# points annotation
"""
positions = vstack([x, y])
savetxt("probe.txt", positions, delimiter=',')
print systemEnergy(positions),  systemEnergy(positions) / len(x)
plt.scatter(x, y, edgecolors = 'none', s = 50, alpha = 0.6)

dx = 0.05
dy = 0.0
for i, xy in enumerate(zip(x, y)):
    plt.axes().annotate(i, (xy[0]+dx, xy[1]+dy), textcoords='data', size = 8)

print "density: ", positions[0].size * pi * diskRadius ** 2 /\
      (max(positions[0]) - min(positions[0])) * (max(positions[1]) - min(positions[1]))

"""

# comparing of surface and volume energy
"""

side = 1.0
drawSquare(side)

side = 0.8
drawSquare(side)
groupOut = []
for i, point in enumerate(positions.T):
    if abs(point[0]) > side or abs(point[1]) > side:
        groupOut.append(i)
print groupOut

side = 0.4
drawSquare(side, 'r')
groupIn = []
for i, point in enumerate(positions.T):
    if abs(point[0]) < side and abs(point[1]) < side:
        groupIn.append(i)
print groupIn

surface = particlesEnergy(positions, groupOut)
volume = particlesEnergy(positions, groupIn)
print "s/f: ", mean(surface.values()) / mean(volume.values())

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

"""

def mu(positions, idx):
    part_energy = 0.0
    vx, vy = 0.0, 1.0
    pc = vstack([positions[0][idx], positions[1][idx]])
    modules = linalg.norm(positions - pc, axis = 0)
    modules[modules==0.0] = inf
    angles = ((positions[0] - pc[0]) * vx + (positions[1] - pc[1]) * vy) / modules

    #print modules
    #print angles
    return sum(sum((35.0 * angles ** 4 - 30.0 * angles ** 2 + 3.0) / (modules ** 5)))

def mu_test(positions, idx):
    vx, vy = 0.5, 0.5

    pc1 = vstack([positions[0][idx], positions[1][idx]])
    modules1 = linalg.norm(positions - pc1, axis = 0)
    modules1[modules1==0.0] = inf
    angles1 = ((positions[0] - pc1[0]) * vx + (positions[1] - pc1[1]) * vy) / modules1
    print

    vec2 = [[vx], [vy]]
    pc2 = positions[:,idx].reshape(2,1)
    modules2 = linalg.norm(positions - pc2, axis = 0)
    modules2[modules2 == 0.0] = inf
    if modules2[modules2 < config.diskRadius * 2].size > 0: return inf
    angles2 = sum((positions - pc2) * vec2, axis=0) / modules2

    print sum((35.0 * angles1 ** 4 - 30.0 * angles1 ** 2 + 3.0) / (modules1 ** 5)), \
        sum((35.0 * angles2 ** 4 - 30.0 * angles2 ** 2 + 3.0) / (modules2 ** 5))
    #print modules1==modules2
    print angles1==angles2

targetDirectory = os.path.dirname(os.path.realpath(__file__)) + "/probe/"

#x, y = dcLattice()
#x, y = reLattice2()
x, y = reLattice()
#x, y = roLattice()
#x, y = diagLattice()
#x, y = pairChainLattice()

x = asarray(x) - 1.0
y = asarray(y) - 1.0
positions = vstack([x, y])
"""
target_idx = 5

en1 = systemEnergy(positions)
mu1 = mu(positions, target_idx)

positions[0][target_idx] += 0.1
positions[1][target_idx] += 0.1

mu2 = mu(positions, target_idx)
print mu_test(positions, target_idx)
en2 = systemEnergy(positions)
print en1, en2, en1 - mu1 + mu2
"""
#chainDelta(positions)
show(positions)
#positions.T[chainIdx * chainLen: chainIdx * chainLen + chainLen]


#plt.plot(positions[0], positions[1], 'o', markeredgecolor='none')
#inColour(x, y)

#show(positions)

#plt.xlim([-0.2, 2.2])
#plt.ylim([-0.2, 2.2])

#indexDiv = [j / len(x) for j in range(len(x) ** 2)]
#indexMod = [j % len(x) for j in range(len(x) ** 2)]

#positions = vstack([x, y]) - 1.0
#print systemEnergy(positions),  systemEnergy(positions) / len(x)

#savetxt(targetDirectory + "probe.txt", positions, delimiter=',')


#savetxt("../data/probe/probe.txt", positions, delimiter=',')
#print len(x)
#plt.plot(x, y, "o")
#plt.show()

