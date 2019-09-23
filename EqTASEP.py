from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import numpy as np
import time
import sys


class EqTASEP:

    def __init__(self, particle_count=100, lattice_size=200, switch_time=100):
        self.lattice_size = lattice_size  # record the size of lattice
        self.particle_count = particle_count  # record number of particles
        self.total_time = 0  # total time taken for the dynamics
        self.forward_time = 0  # record time taken for forward dynamics
        self.switch_time = switch_time  # seconds to switch from forward to equil jump
        self.interval = 1  # average jump interval
        # lattice structure to store position of particles and holes
        self.lattice = np.zeros(lattice_size, int)
        # array of timers for all particles
        self.particle_clock = np.zeros(lattice_size)
        # array of timers for all holes
        self.hole_clock = np.zeros(lattice_size)
        # time interval during the jump
        self.jump_interval = self.particle_count / 400  # i.e. give 400 points to plot
        self.total_jump = 0  # total no of jumps taken in the process

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

    def __normalizeHoleClock(self, clock_array):
        """
            Normalize all timers in the original hole_clock by generating new timers for negative values after updating the clock.

            Parameters
            ----------
            clock_array: np array
                Array of timers that is to be normalized after updating the timer.
        """
        for i in range(len(clock_array)):
            if clock_array[i] < 0:  # i.e. negative time after update
                # find no of particle to the right
                n = self.__findParticleToRight(i)

                # avoid devision by 0, occurs when the last spot is empty
                if n != 0:
                    # generate a random time with mean t/n
                    t = np.random.exponential(self.interval*self.switch_time/n)
                    # record that time in clock_array
                    clock_array[i] = t

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

    def __particleMinTime(self, moving_particles):
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
        clock = self.particle_clock[moving_particles]

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
        # loop over the array to assign new timer at proper position (initial value is 0)
        for i in range(self.lattice_size):

            if self.lattice[i] == 0:  # if a slot is empty
                # find no of particle to the right
                n = self.__findParticleToRight(i)

                # avoid devision by 0, occurs when the last spot is empty
                if n != 0:
                    # generate a random time with mean t/n
                    t = np.random.exponential(self.interval*self.switch_time/n)
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

    def jumpForward(self):
        """
            Update the original lattice structure and clock only after a particle jumping forward. The edges are fixed.
        """
        # find and record
        # find a list of indices of movable particles
        moving_particles = self.__findMovableParticles()
        # find the position of clock with min time in the lattice, and the min time
        position, min_time = self.__particleMinTime(moving_particles)

        # update and normalize the clock after jump
        self.__updateClock(min_time, self.particle_clock)
        self.__updateClock(min_time, self.hole_clock)
        self.__normalizeClock(self.particle_clock)
        self.__normalizeHoleClock(self.hole_clock)

        # update the lattice accordingly
        # allow the particle to jump forward if it is not at the end of the lattice
        if position < self.lattice_size - 1:

            # before jump
            self.lattice[position] = 0  # empty current spot
            # create an exponential random time for the current position (a hole after jump) with proper mean t/n
            n = self.__findParticleToRight(position)+1
            t1 = np.random.exponential(self.interval*self.switch_time/n)

            # reset timer for current hole
            self.hole_clock[position] = t1
            self.particle_clock[position] = 0

            # jump forward
            position += 1  # increment the postion index
            self.lattice[position] = 1  # occupy the spot after jump
            # create an exponential random time for the particle at its new position
            t2 = np.random.exponential(self.interval)

            # reset timer for current particle
            self.particle_clock[position] = t2
            self.hole_clock[position] = 0

        # do not allow the particle to exit from the right if it is at the end of the lattice
        # do not allow a new particle to enter the lattice from the left

        self.total_time += min_time  # record the total time taken
        self.forward_time += min_time  # record the time taken for forward dynamics

    def jumpBackward(self):
        """
            Update the original lattice by allowing particles to jump backwards where each hole trys to attract its nearest particle to the right with some given rate.
        """
        # find the position of clock with min time in the lattice, and the min time
        position, min_time = self.__holeMinTime()

        # update and normalize the clock after jump
        self.__updateClock(min_time, self.particle_clock)
        self.__updateClock(min_time, self.hole_clock)
        self.__normalizeClock(self.particle_clock)
        self.__normalizeHoleClock(self.hole_clock)

        # slicing the original lattice to start from the index with min time
        temp_lattice = self.lattice[position:]
        # find the index of nearest particle to the right, particle position
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

        # reset time within index between position+1 and p_position (inclusive) only
        for i in range(position + 1, p_position + 1):

            # initialize the clock as 0 first
            self.hole_clock[i] = 0
            # find no of particle to the right
            n = self.__findParticleToRight(i)

            # avoid devision by 0, occurs when the last spot is empty
            if n != 0:

                # generate a random time with mean t/n
                t2 = np.random.exponential(self.interval*self.switch_time/n)
                # record that time in clock_array
                self.hole_clock[i] = t2

        self.total_time += min_time  # decrese the total time taken
        # do not minus here

    def jumpEquil(self):
        """
            Update the original lattice by allowing particles to jump in equilibrium with forward and backward dynamics in adjusted rate.
        """
        # find the smallest timer in the lattice for hole and particle
        moving_particles = self.__findMovableParticles()
        p_min_time = np.min(self.particle_clock[moving_particles])
        h_min_time = np.min(self.hole_clock[np.nonzero(self.hole_clock)])

        # jump backward if the timer for a hole is smaller
        if h_min_time <= p_min_time:
            self.jumpBackward()

        # jump forward if the timer for a particle is smaller
        elif p_min_time < h_min_time:
            self.jumpForward()

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

    def writeToFile(self, file_name="eq_tasep.txt"):
        """
            Allow TASEP to run and write information obtained from the process to a local text file.

            Parameters
            ----------
            file_name: string
                The name of the file that data are written to. Format is txt.
        """
        with open(file_name, "w") as text_file:

            # forward jump until the switching time (= no. of particle/2 seconds)
            while self.forward_time <= self.switch_time:
                self.jumpForward()

                # print timestamp and the height to the local file
                if self.total_jump <= self.total_time:
                    print((self.total_jump, self.heightFunc(self.lattice)[
                        self.particle_count]), file=text_file)

                    self.total_jump += self.jump_interval  # increment for the next recording

            # equil jump for a specified amount of time (twice the forward)
            while self.total_time <= self.switch_time*3:
                self.jumpEquil()  # allow the particles to jump in equilibrium

                # print timestamp and the height to the local file
                if self.total_jump <= self.total_time:
                    print((self.total_jump, self.heightFunc(self.lattice)[
                        self.particle_count]), file=text_file)

                    self.total_jump += self.jump_interval  # increment for the next recording

    def plotting(self, file_name="eq_tasep.txt", fig_name="eq_tasep.png"):
        """
            Plot the data of height against timestamp and save the plotting.

            Parameters
            ----------
            file_name: string
                Name of the file with data for plotting. Format is txt.

            fig_name: string
                Name of the figure to be saved. Format is png.
        """
        # x (time), y (height) values
        x = []
        y = []

        # read the file and get (x, y) value pairs
        with open(file_name, "r") as f:

            # record x,y values for each line
            for line in f.readlines():
                # format the string that's printed
                data = line.strip("(").strip(")\n").split(", ")
                # coerce to integer
                x.append(int(float(data[0])))
                y.append(int(data[1]))

        # plots setting
        plt.title('Equilibrium TASEP')
        plt.xlabel("Timestamp")
        plt.ylabel("Height")
        plt.plot(x, y, "-")
        plt.savefig(fig_name)
        plt.show()


# total amount of particle in the simulation, key input
particle_count = 1000  # try 1000000 particles

# initialize data input
data = EqTASEP(
    particle_count=particle_count,
    lattice_size=2*particle_count,
    switch_time=particle_count/2
)
data.buildStepIC()  # build the lattice structure and its clock
data.writeToFile()  # run the process and record data to a file
data.plotting()  # plot and save the process
