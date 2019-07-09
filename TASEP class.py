"""
Forward and reverse TASEP simulation

author: Haoyu Li, email: hl6de@virginia.edu
guided by prof Leonid Petrov
"""


from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import numpy as np
import time
import sys


class TASEP:

    def __init__(self, particle_count=100, lattice_size=200):
        self.lattice_size = lattice_size  # record the size of lattice
        self.particle_count = particle_count  # record number of particles
        self.total_time = 0  # total time taken for the dynamics
        self.interval = 1  # average jump interval
        # lattice structure to store position of particles and holes
        self.lattice = np.zeros(lattice_size, int)
        # array of timers for all particles
        self.particle_clock = np.zeros(lattice_size)
        # array of timers for all holes
        self.hole_clock = np.zeros(lattice_size)
        self.initial_structure = []  # initial structure used for comparision
        self.forward_time = 0  # record time taken for forward dynamics

    def __findParticleToRight(self, index):
        """
            Calculate the number of particles to the right of a given positional index in the lattice.

            Parameters
            ----------
            index : int
                Index of current position in the lattice.

            Returns
            -------
            num : int
                The number of particles to the right of current index postion.
        """
        # slice the lattice starting from the given index
        temp_lattice = self.lattice[index:]
        # count the spots with value = 1, i.e. occupied
        num = np.count_nonzero(temp_lattice)

        return num

    def __updateClock(self, min_time, clock_array):
        """
            Update the clock by decreasing all valid (non-zero) values by the min_time. Change is made inside the array.

            Parameters
            ----------
            min_time : float
                Minimum time taken for a particle to jump in the lattice.

            clock_array: nparray
                Array of timers that is to be updated after every jump.
        """
        np.subtract(clock_array, min_time,
                    out=clock_array, where=clock_array > 0)

    def __normalizeClock(self, clock_array):
        """
            Normalize all timers in the original particle_clock by generating new timers for negative values after updating the clock.

            Parameters
            ----------
            clock_array: np array
                Array of timers that is to be normalized after updating the timer.
        """
        # length/amount of negative timers
        neg_len = len(clock_array[clock_array < 0])
        # keep positive timers, replace negative timers with new exponentially generated random timer
        clock_array[clock_array < 0] = np.random.exponential(1, neg_len)

    def __findMovableParticles(self):
        """
            Find particles in the lattice that can jump forward with the next spot being empty.

            Returns
            -------
            movingParticles : list <int> 
                A list of indexes of particles that can jump forward.
        """
        # a list to store positional index of particles that can possibly move forward
        moving_particles = []

        # excluding the last spot, which will stay there
        for i in range(self.lattice_size-1):

            # find the index in the lattice with current spot occupied and the next empty
            if self.lattice[i] == 1 and self.lattice[i+1] == 0:
                moving_particles.append(i)  # record that positional index

        return moving_particles

    def __particleMinTime(self, movingParticles):
        """
            Get an array of timer from particle_clock for particles with indices provided in the list, find the min time for those potentially moving particles.

            Parameters
            ----------
            movingParticles : list <int> 
                A list of indexes of particles that can jump forward.

            Returns
            -------
            minid, minval : (int, float) touple
                The positional index  of the minimal value and the minimal value in the clock_array.
        """
        # find the corresponding timer for particles with indices provided
        clock = self.particle_clock[movingParticles]

        # find the min time and its index in the clock array
        minval = np.min(clock)
        minid = np.where(self.particle_clock == minval)[0][0]

        return minid, minval

    def __holeMinTime(self):
        """
            Find a valid (non-zero) min time and its index for holes in hole_clock.

            Returns
            -------
            minid, minval : (int, float) touple
                The positional index  of the minimal value and the minimal value in the clock_array.
        """
        # min value of the lattice
        minval = np.min(self.hole_clock[np.nonzero(self.hole_clock)])
        # index of the min in the array
        minid = np.where(self.hole_clock == minval)[0][0]

        return minid, minval

    def __runParticleClock(self):
        """
            Assign an independent random exponential clock to each particle in the lattice after initialize the lattice structure. Assign 0 to all the holes.
        """
        for i in range(self.lattice_size):

            # if a particle occupies the slot
            if self.lattice[i] == 1:
                # generate a random time with mean 1
                t = np.random.exponential(self.interval)
                # record that time in clock_array
                self.particle_clock[i] = t

    def __runHoleClock(self):
        """
            Assign an independent random exponential clock to each hole in the lattice with rate multipled by no. of partiles to the right of it. Assign 0 to all particles.
        """
        # loop over the array to assign new timer at proper position
        for i in range(self.lattice_size):

            if self.lattice[i] == 0:  # if a slot is empty
                # find no of particle to the right
                n = self.__findParticleToRight(i)

                # avoid devision by 0, occurs when the last spot is empty
                if n != 0:
                    # generate a random time with mean 1/n
                    t = np.random.exponential(self.interval/n)
                    # record that time in clock_array
                    self.hole_clock[i] = t

    def buildStepIC(self):
        """
            Build a Step IC lattice structure with specific amount of particles and specific lattice size. All particle at the left of the lattice and all holes are at the right of the lattice. Record the initial lattice structure and time.
        """
        # fill the slot occupied by particles with 1 (= occupied) from the left
        for i in range(self.particle_count):
            self.lattice[i] = 1

        self.initial_structure = self.lattice.copy()  # record initial structure
        self.__runParticleClock()  # randomly generated time for each particle
        self.__runHoleClock()  # randomly generated time for each hole

        # draw the initial plot first as reference to the original shape and second to automatically get a better axis range
        self.displayPlot(self.initial_structure)

    def jumpForward(self):
        """
            Update the original lattice structure and clock only after a particle jumping forward. The edges are fixed.
        """
        # find and record
        # find a list of indices of movable particles
        moving_particles = self.__findMovableParticles()
        # find the position of clock with min time in the lattice, and the min time
        position, min_time = self.__particleMinTime(moving_particles)

        self.total_time += min_time  # record the total time taken
        self.forward_time += min_time  # record the time taken for forward dynamics

        # update the clock after jump
        self.__updateClock(min_time, self.particle_clock)
        # normalize each particle clock value
        self.__normalizeClock(self.particle_clock)

        # update the lattice accordingly
        # allow the particle to jump forward if it is not at the end of the lattice
        if position < self.lattice_size - 1:

            # before jump
            self.lattice[position] = 0  # empty current spot
            # create an exponential random time for the current position (a hole after jump)
            t1 = np.random.exponential(self.interval)
            self.hole_clock[position] = t1  # reset timer

            # jump forward
            position += 1  # increment the postion index
            self.lattice[position] = 1  # occupy the spot after jump
            # create another exponential random time for the particle at its new position
            t2 = np.random.exponential(self.interval)
            self.particle_clock[position] = t2  # reset timer
            # empty the clock of holes at the new particle spot
            self.hole_clock[position] = 0

        # do not allow the particle to exit from the right if it is at the end of the lattice
        # do not allow a new particle to enter the lattice from the left

    def jumpBackward(self):
        """
            Update the original lattice by allowing particles to jump backwards where each hole trys to attract its nearest particle to the right with some given rate.
        """
        # if the lattice already return to the original structure i.e. all particles are to the left
        if np.array_equal(self.lattice, self.initial_structure):
            exit()  # end the program

        # find the position of clock with min time in the lattice, and the min time
        position, min_time = self.__holeMinTime()
        # update the clock_array by subtracting min time from each clock
        self.__updateClock(min_time, self.hole_clock)

        # slicing the original lattice to start from the index with min time
        temp_lattice = self.lattice[position:]
        # find the index of nearest particle to the right, pposition
        p_position = position + np.where(temp_lattice == 1)[0][0]

        # reset the lattice structure after the reverse jump
        self.lattice[position] = 1  # now the hole is occupied
        # create an exponential random time for the particle after occupying the hole
        t1 = np.random.exponential(self.interval)
        self.particle_clock[position] = t1

        # the nearest right particle leaves its original position
        self.lattice[p_position] = 0  # now the particle slot is emptied
        # empty the clock of holes at the old particle spot (now it is a hole)
        self.particle_clock[p_position] = 0

        # reset time within index between position+1 and p_position only
        for i in range(position + 1, p_position + 1):

            # initialize the clock as 0 first
            self.hole_clock[i] = 0
            # find no of particle to the right
            n = self.__findParticleToRight(i)

            # avoid devision by 0, occurs when the last spot is empty
            if n != 0:

                # generate a random time with mean 1/n
                t2 = np.random.exponential(self.interval/n)
                # record that time in clock_array
                self.hole_clock[i] = t2

        self.total_time /= np.exp(min_time)  # decrese the total time taken

    def heightFunc(self, structure):
        """
            Build a height function of particles in the lattice, given the structure.

            Parameters
            ----------
            structure: nparray
                The lattice structure to be plotted as height function.

            Returns
            -------
            y : nparray <int>
                The height of lattice structure at each index.
        """
        # create a copy to avoid changing values in original lattice
        temp_lattice = self.lattice.copy()

        # normalize the temp lattice for cumsum function
        # i.e. change 0 (empty slot) to 1, change 1 (occupied) to -1
        temp_lattice = 1 - 2 * temp_lattice

        # insert the total particle count at the front to build proper shape above x-axis
        # using concatenate is faster than insert
        temp_lattice = np.concatenate(([self.particle_count], temp_lattice))

        # cumsum helps to accumulate values in the array and output an array
        y = np.cumsum(temp_lattice)

        return y

    def limitHeightFunction(self, x):
        """
            Build a height function of particles in the limiting case, i.e. h = t(1+(x/t)^2)/2 where t is time and x is position.

            Parameters
            ----------
            x : float
                Input x value

            Returns
            -------
            y : nparray <int>
                The output height of lattice in the limiting case
        """
        # height function of TASEP in limiting case
        y = self.total_time*(((x-self.lattice_size/2)/self.total_time)**2+1)/2
        return y

    def locationFunc(self):
        """
            Mark the locations of particles on x-axis, given the structure of particle in the lattice.

            Returns
            -------
            x : nparray <int>
                The positional indexes of each particle on x-axis
        """
        # indeces of occupied particles in the lattice, i.e. value = 1
        x = np.nonzero(self.lattice)
        return x

    def displayPlot(self, structure):
        """
            Display the initial graph given lattice data.
        """
        # initial height value pair
        y1 = self.heightFunc(structure)
        x1 = np.arange(len(y1))

        plt.plot(x1, y1, "y-", label="initial state")


# ------------------------------------------------------------
#
# animation
#


# real time element to be displayed
dt = 1./30  # 30 fps
t0 = time.time()
t1 = time.time()
interval = 1000 * dt - (t1 - t0)


# set up the plot, figure and axis
fig = plt.figure()
ax = plt.axes()

# initialize three lines to be plotted
particle_line, = ax.plot([], [], "r.", label="particle")
real_height, = ax.plot([], [], "c-", label="real-time height")
limit_height, = ax.plot([], [], "g-", label="limit-case height")
# text area for time
time_text = ax.text(0.05, 0.1, '', transform=ax.transAxes)

# initialize data input
data = TASEP(particle_count=400, lattice_size=800)
data.buildStepIC()  # build the lattice structure

# amount of forward time in update at which the process is switched
# can change if want longer forward animation
switch_time = data.particle_count/2


# set up the style and axes for the plot
def setup():
    
    mplstyle.use('fast')  # to make the plotting faster
    ax.legend(loc=9)  # show legend at upper center
    
    # hide x, y label
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.title('TASEP Simulator')  # display title


# initial graph with 3 lines and text to be updated
def init():

    # get new (x1, y1), (x2, y2) value pair
    y1 = data.heightFunc(data.lattice)
    # all points on x-axis, this is fixed, no need updating
    x1 = np.arange(len(y1))

    x2 = data.locationFunc()
    # no. of particles in the lattice, this is fixed, no need updating
    y2 = np.zeros(len(x2), int)

    # set new data to each lines to be drawn
    real_height.set_data(x1, y1)
    particle_line.set_data(x2, y2)

    time_text.set_text('')
    return real_height, particle_line, limit_height, time_text


# forward/backward jump animation, update the graph after potential movement
def update(i):

    # allow only one direction TASEP
    if data.forward_time <= switch_time:
        data.jumpForward()  # allow the particles to jump forward

    else:
        data.jumpBackward()  # allow the particles to jump backward

    # get new (x1, y1), (x2, y2), (x3, y3) value pair
    y1 = data.heightFunc(data.lattice)
    x2 = data.locationFunc()

    # set new data to each lines to be drawn
    real_height.set_ydata(y1)
    particle_line.set_xdata(x2)

    # x3 is in [-T,T] about the center of the graph
    x3 = np.arange(data.lattice_size/2-data.total_time,
                   data.lattice_size/2+data.total_time)
    y3 = data.limitHeightFunction(x3)
    limit_height.set_data(x3, y3)

    time_text.set_text('time = %.5f' % data.total_time)

    return real_height, particle_line, limit_height, time_text


# set up the plot before animating
setup()

# animate the tasep process
ani = FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=interval,
    blit=True,
    repeat=False,
    save_count=sys.maxsize
)

# save the the animation
# ani.save('tasep_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# show the plotting and animation
plt.show()
