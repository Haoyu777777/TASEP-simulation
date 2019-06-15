"""
Forward and reverse TASEP simulation

author: Haoyu Li, email: hl6de@virginia.edu
guided by prof Leonid Petrov
"""


from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import time
import os


class TASEP:

    def __init__(self, particle_count=100, lattice_size=200):
        self.lattice_size = lattice_size
        self.particle_count = particle_count
        self.total_time = 0
        self.interval = 1
        self.lattice = []
        self.particle_clock = []
        self.hole_clock = []
        self.time_list = []
        self.initial_structure = []

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
        tempLattice = self.lattice[index:]
        # count the spots with value = 1, i.e. occupied
        num = np.count_nonzero(tempLattice)

        return num

    def __updateParticleClock(self, minTime):
        """
        Update the particle_clock by decreasing all valid (non-zero) values by the minTime. Change is made inside the array.

        Parameters
        ----------
        minTime : float
            Minimum time taken for a particle to jump in the lattice.

        """

        np.subtract(self.particle_clock, minTime,
                    out=self.particle_clock, where=self.particle_clock > 0)

    def __updateHoleClock(self, minTime):
        """
        Update the hole_clock by decreasing all valid (non-zero) values by the minTime. Change is made inside the array.

        Parameters
        ----------
        minTime : float
            Minimum time taken for a particle to jump in the lattice.

        """

        np.subtract(self.hole_clock, minTime, out=self.hole_clock,
                    where=self.hole_clock > 0)

    def __normalizeParticleClock(self):
        """
        Normalize all timers in the original particle_clock by generating new timers for negative values after updating the clock.
        """

        # amount of negative times
        neg = len(self.particle_clock[self.particle_clock < 0])

        # keep positive timer, replace negative timer with new exponentially generated random timer
        self.particle_clock[self.particle_clock <
                            0] = np.random.exponential(1, neg)

    def __findMovableParticles(self):
        """
        Find particles in the lattice that can jump forward with the next spot being empty.

        Returns
        -------
        movingParticles : list <int> 
            A list of indexes of particles that can jump forward.
        """

        # a list to store positional index of particles that can possibly move forward
        movingParticles = []

        # excluding the last spot to handle differently (which will leave the lattice directly)
        for i in range(self.lattice_size-1):

            # find the index in the lattice with current spot occupied and the next empty
            if self.lattice[i] == 1 and self.lattice[i+1] == 0:
                movingParticles.append(i)  # record that positional index

        return movingParticles

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

    def runParticleClock(self):
        """
            Assign an independent random exponential clock to each particle in the lattice after initialize the lattice structure. Assign 0 to all the holes.
        """
        # initialize an array with proper size to store random clock
        self.particle_clock = np.zeros(self.lattice_size)

        for i in range(self.lattice_size):

           # if a particle occupies the slot
            if self.lattice[i] == 1:

                # generate a random time with mean 1
                t = np.random.exponential(self.interval)
                # record that time in clock_array
                self.particle_clock[i] = t

    def runHoleClock(self):
        """
            Assign an independent random exponential clock to each hole in the lattice with rate multipled by no. of partiles to the right of it. Assign 0 to all particles.

        """
       # initialize an array with proper size to store random clock
        self.hole_clock = np.zeros(self.lattice_size)

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
        # fill all lattice slots with 0 (= empty) first
        self.lattice = np.zeros(self.lattice_size, int)

        # fill the slot occupied by particles with 1 (= occupied) from the left
        for i in range(self.particle_count):
            self.lattice[i] = 1

        self.time_list.append(0)  # record time
        self.initial_structure = self.lattice.copy()  # record initial structure

    def jumpForward(self):
        """
        Update the original lattice structure and clock only after a particle jumping forward. The edges are fixed.
        """

        # find and record
        # find a list of indices of movable particles
        movingParticles = self.__findMovableParticles()
        # find the position of clock with min time in the lattice, and the min time
        position, minTime = self.__particleMinTime(movingParticles)

        self.time_list.append(minTime)  # record time taken for this jump
        self.total_time += minTime  # record the total time taken

        # update the clock after jump
        self.__updateParticleClock(minTime)
        # normalize each clock value
        self.__normalizeParticleClock()

        # update the lattice accordingly
        # allow the particle to jump forward if it is not at the end of the lattice
        if position < self.lattice_size - 1:

            # empty current spot before jump
            self.lattice[position] = 0

            position += 1  # increment the postion index
            self.lattice[position] = 1  # occupy the spot after jump

            # create an exponential random time
            t = np.random.exponential(self.interval)
            self.particle_clock[position] = t  # reset timer

        # do not allow the particle to exit from the right if it is at the end of the lattice
        # do not allow a new particle to enter the lattice from the left

    def jumpBackward(self):
        """
        Update the original lattice by allowing particles to jump backwards where each hole trys to attract its nearest particle to the right with some given rate.
        """

        # if the lattice already return to the original structure
        # i.e. all particles are to the left
        if np.array_equal(self.lattice, self.initial_structure):
            exit()  # end the program

        # find the position of clock with min time in the lattice, and the min time
        position, minTime = self.__holeMinTime()
        # update the clock_array by subtracting min time from each clock
        self.__updateHoleClock(minTime)

        # slicing the original lattice to start from the index with min time
        tempLattice = self.lattice[position:]
        # find the index of nearest particle to the right, pposition
        pposition = position + np.where(tempLattice == 1)[0][0]

        # reset the lattice structure after the reverse jump
        self.lattice[position] = 1  # now the hole is occupied
        # the nearest right particle leaves its original position
        self.lattice[pposition] = 0  # now the particle slot is emptied

        # reset time within index between position+1 and pposition only
        for i in range(position + 1, pposition + 1):

            # initialize the clock as 0 first
            self.hole_clock[i] = 0

            # find no of particle to the right
            n = self.__findParticleToRight(i)

            # avoid devision by 0, occurs when the last spot is empty
            if n != 0:

                # generate a random time with mean 1/n
                t = np.random.exponential(self.interval/n)
                # record that time in clock_array
                self.hole_clock[i] = t

        self.time_list.append(0)  # record time in the list
        self.total_time += minTime  # increment the total time taken

    def heightFunc(self):
        """
        Build a height function of particles in the lattice, given the structure.

        Returns
        -------
        y : nparray <int>
            The height of lattice structure at each index
        """

        # create a copy to avoid changing values in original lattice
        tempLattice = self.lattice.copy()

        # normalize the temp lattice for cumsum function
        # i.e. change 0 (empty slot) to 1, change 1 (occupied) to -1
        tempLattice = 1 - 2 * tempLattice

        # insert the total particle count at the front to build proper shape above x-axis
        # using concatenate is faster than insert
        tempLattice = np.concatenate(([self.particle_count], tempLattice))

        # cumsum helps to accumulate values in the array and output an array
        y = np.cumsum(tempLattice)

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

    def displayPlot(self):
        """
        Display the initial graph given lattice data.

        """
        # height value pair
        y1 = self.heightFunc()
        x1 = np.arange(len(y1))

        # location value pair
        # x2 = self.locationFunc()
        # y2 = np.zeros(len(x2), int)

        # plot graphs based on (x1, y1), (x2, y2) value pairs
        plt.plot(x1, y1)  # plot x1, y1 as blue line (default)
        # plt.plot(x2, y2, "r.")  # plot x2, y2 as red dot


# ------------------------------------------------------------


# animation
# time element to be displayed
dt = 1./30  # 30 fps
t0 = time.time()
t1 = time.time()
interval = 1000 * dt - (t1 - t0)


# set up figure and axis
fig = plt.figure()
ax1 = plt.axes()
plt.title('TASEP Simulator')

# hide x, y label
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)

# initialize and store two lines to be plotted
lines = []
line1, = ax1.plot(0, 0)
line2, = ax1.plot(0, 0, "r.")
lines.append(line1)
lines.append(line2)
# text area for time
time_text = ax1.text(0.05, 0.1, '', transform=ax1.transAxes)


# initialize data input
data = TASEP(particle_count=400, lattice_size=800)
data.buildStepIC()  # build the lattice structure
data.runParticleClock()  # randomly generated time for each particle

# wait for 300s' forward jump (for backward simulation)
while data.total_time < 300:
    data.jumpForward()
data.runHoleClock()  # randomly generated time for each hole


# initial graph with 2 lines to be updated
def init():

    # get new (x1, y1), (x2, y2) value pair
    y1 = data.heightFunc()
    # all points on x-axis, this is fixed, no need updating
    x1 = np.arange(len(y1))

    x2 = data.locationFunc()
    # no. of particles in the lattice, this is fixed, no need updating
    y2 = np.zeros(len(x2), int)

    # set new data to each lines to be drawn
    line1.set_data(x1, y1)
    line2.set_data(x2, y2)

    time_text.set_text('')

    return lines, time_text


# forward/backward jump animation
# update the graph after potential movement
def update(i):

    # allow only one direction TASEP
    # data.jumpForward()  # allow the particles to jump forward
    data.jumpBackward()  # allow the particles to jump backward

    # get new (x1, y1), (x2, y2) value pair
    y1 = data.heightFunc()
    x2 = data.locationFunc()

    # set new data to each lines to be drawn
    line1.set_ydata(y1)
    line2.set_xdata(x2)

    time_text.set_text('time = %.5f' % data.total_time)

    return lines, time_text


# draw the initial plot for reference
data.displayPlot()

# animation, interval set to 1 to avoid delay
ani1 = FuncAnimation(fig, update, frames=data.time_list,
                     init_func=init, interval=interval)

# show the plotting and animation
plt.show()
