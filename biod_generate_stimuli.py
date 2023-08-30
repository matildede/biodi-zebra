import pandas as pd
import numpy as np
import math

np.random.seed(18)

class biod:
    def __init__(self, x_max, y_max, maxSpeed=16):

        # Euclidean vector with random x y position within a window (width*height)
        self.location = np.array([np.random.randint(x_max), np.random.randint(y_max)])

        # creates a random vector features: magnitude(8-16) and direction (angle in radians 0-2π (360 degrees)
        self.velocity_feature = [np.random.randint(8, 16), np.radians(np.random.randint(0, 360))]

        # Calculate the components of the vector from the magnitude and direction
        # cos and sin functions are used to calculate the x and y components of the vector, respectively
        self.velocity = np.array([self.velocity_feature[0] * np.cos(self.velocity_feature[1]),
                                  self.velocity_feature[0] * np.sin(self.velocity_feature[1])])

        # initial null acceleration
        self.acceleration = np.array([0, 0])

        # limit the magnitude steering force.
        self.maxForce = 2

        # limit the magnitude of velocity, i.e. speed.
        self.maxSpeed = maxSpeed

    def random(self, magnitude):
        # Generate a random acceleration vector
        # Calculate the components of the acceleration vector from its magnitude and direction
        randian = np.radians(np.random.randint(0, 360))
        self.acceleration = np.array([magnitude * np.cos(randian), magnitude * np.sin(randian)])

    def flock(self, biods):
        """
        This method compute the three different steering forces.
        Then it scales each of them and adds them together.

        :param biods:
        """
        alignment = self.align(biods)
        cohesion = self.cohesion(biods)
        separation = self.separation(biods)

        scaling_factor = (maxSpeed/2)/8

        # Scale the three steering forces and add them together
        alignment = alignment * 1 * scaling_factor  # 1
        cohesion = cohesion * 0.4 * scaling_factor # 0.6 (6), 0.5 (3) 1.1 or 2.4
        separation = separation * 2.5 * scaling_factor # 1.6 or 2.5

        self.acceleration = self.acceleration + alignment + cohesion + separation

    def align(self, biods):
        """
        This method calculates the average velocity of all the other biods and set it as the desired.
        Indeed, this average vector indicates the mean direction of all the other boids
        """
        perception = 100
        tot_velocity = np.array([0, 0])
        count = 0
        steer = np.array([0, 0])

        for other in biods:
            d = np.linalg.norm(self.location - other.location)  # magnitude of the distance between the two points
            if other is not self and d < perception:
                tot_velocity = tot_velocity + other.velocity  # add all the velocities
                # print(tot_velocity)
                # print(other.velocity)
                count += 1

        if count > 0:
            # print(count)
            average_velocity = tot_velocity / count  # calculate desired velocity = average velocity of other boids
            # print(desired_velocity)
            desired_velocity = (average_velocity / np.linalg.norm(
                average_velocity)) * self.maxSpeed  # normalize and set the magnitude of the vector to maxSpeed
            # print(desired_velocity)
            steer = desired_velocity - self.velocity  # calculates the steering force
            # in base a cosa hai scelto la max force?
            if np.linalg.norm(steer) < self.maxForce:
                pass
            else:
                # limit the magnitude but maintain the direction of the vector:
                direction = math.atan2(steer[1], steer[0]) / math.pi * 180  # non capisco bene cosa faccia qui
                # print(f'x = {steer[0]}  y = {steer[1]}  direction = {direction}')
                steer[0] = self.maxForce * np.cos(np.radians(direction))
                steer[1] = self.maxForce * np.sin(np.radians(direction))
                # print(steer[0])
                # print(steer[1])

            # print(f'steer = {steer}')

        # print(f'count = {count}')
        # print(f'steer = {steer}')
        return steer

    def cohesion(self, biods):
        """
        This method calculates the average location of the biod's neighbors and set it as the target to seek.
        According to it, it calculates desired velocity and steering force.
        :param biods:
        :return:
        """
        perception = 500
        tot_location = np.array([0, 0])
        count = 0
        steer = np.array([0, 0])

        for other in biods:
            d = np.linalg.norm(self.location - other.location)
            if other is not self and d < perception:
                # add all the others' locations
                tot_location = tot_location + other.location
                count += 1
        if count > 0:
            # average location that will be used for calculating desired velocity
            average_location = tot_location / count
            # calculate desired velocity = distance between self.location and average location of other boids
            desired_velocity = average_location - self.location
            # normalize and set the magnitude of the vector to maxSpeed
            desired_velocity = (desired_velocity / np.linalg.norm(desired_velocity)) * self.maxSpeed
            steer = desired_velocity - self.velocity  # calculates the steering force
            if np.linalg.norm(steer) < self.maxForce:
                pass
            else:
                # limit the magnitude of the steer vector within the range +- maxForcelimit but maintain the direction
                direction = math.atan2(steer[1], steer[0]) / math.pi * 180  # non capisco bene cosa faccia qui
                steer[0] = self.maxForce * np.cos(np.radians(direction))
                steer[1] = self.maxForce * np.sin(np.radians(direction))
        return steer

    def separation(self, biods):
        """
        This method calculates the steering force to be applied to each biod in the shoal
        in order to avoid colliding with its neighbors.
        :param self: biod for which the steering force is calculated
        :param biods: all the other biod objects in the shoal with their properties
        :return: steering force
        """
        perception = 100
        tot_distance = np.array([0, 0])
        count = 0
        steer = np.array([0, 0])

        for other in biods:
            d = np.linalg.norm(self.location - other.location)
            if other is not self and d < perception:
                # calculate desired velocity as a vector pointing away from the other's location
                distance = self.location - other.location
                # normalize the distance vector
                distance = distance / np.linalg.norm(distance)
                # add the direction for the desired velocity with respect to each other biod
                tot_distance = tot_distance + distance
                count += 1
        if count > 0:
            # average of the desired velocity
            desired_velocity = tot_distance / count
            # normalize and set the magnitude of the vector to maxSpeed
            desired_velocity = (desired_velocity / np.linalg.norm(desired_velocity)) * self.maxSpeed
            steer = desired_velocity - self.velocity  # calculate the steering force
            if np.linalg.norm(steer) < self.maxForce:
                pass
            else:
                # limit the magnitude of the steer vector within the range +- maxForcelimit but maintain the direction
                direction = math.atan2(steer[1], steer[0]) / math.pi * 180  # non capisco bene cosa faccia qui
                steer[0] = self.maxForce * np.cos(np.radians(direction))
                steer[1] = self.maxForce * np.sin(np.radians(direction))
        return steer

    def update(self):
        # changing the order of each line changes the behaviour of the biods
        # for example velocity,location,acceleration creates problems for the borders
        """
        This method updates the location on the basis of the computed velocity and update velocity with acceleration.
        Also, it resets acceleration each time. It's not clear to me if this order has to be mantained or if location
        can be computed after velocity.
        """

        # change location by velocity
        self.location = self.location + self.velocity

        self.velocity = self.velocity + self.acceleration

        # rescale and set the magnitude of the vector
        self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * (self.maxSpeed / 2)

        # check magnitude of speed vector
        speed = np.linalg.norm(self.velocity)

        # We reset the forces, i.e. acceleration to zero. This is needed to avoid summing forces from t to t+1
        self.acceleration = self.acceleration * 0
        return speed

    def borders(self, x_max, y_max, limit):
        """
        This method avoid biods to exit from the window and make them bounce on the borders
        """
        if self.location[0] < limit and self.velocity[0] < 0:
            self.velocity[0] = self.velocity[0] * -1
        elif self.location[0] > x_max - limit and self.velocity[0] > 0:
            self.velocity[0] = self.velocity[0] * -1
        if self.location[1] < limit and self.velocity[1] < 0:
            self.velocity[1] = self.velocity[1] * -1
        elif self.location[1] > y_max - limit and self.velocity[1] > 0:
            self.velocity[1] = self.velocity[1] * -1

    def random2(self, magnitude):
        # Generate a random acceleration vector
        # Calculate the components of the acceleration vector from its magnitude and direction
        self.acceleration = np.array([1, 0])

if __name__ == '__main__':

    experimental_group = 'flock'  # 'flock', 'random1', 'random2', 'random3'
    n_biods = 3  #3 or 6
    maxSpeed = 6  # 6, 12, 24
    minutes = 5
    framerate = 30
    time_points = minutes * framerate * 60 + 10  # + 10 so to have a slightly bigger dataframe and avoid crashing of psychopy

    shoal = []
    x_max = 700  # width of my window
    y_max = 700  # height of my window

    # Here I initialize my objects. In this way I attribute to each of them one of the class biod attributes
    # eg. location, velocity, acceleration etc.

    for i in range(n_biods):
        shoal.append(biod(x_max, y_max, maxSpeed))
        if experimental_group == 'random3':
            shoal[0].location = np.array([0, 450])
            shoal[0].velocity = np.array([1, 0])


    # Here I update the positions of my biods and apply the forces defined in flock().
    # I also create a big list with the positions in x and y of all my biods for each frame
    # and for each time point. I also create a list where I specify the identity of each biod
    # in order to check later if I made some mistake.


    time = []

    for i, biod in enumerate(shoal):
        exec(f"biod{i}_x = []")
        exec(f"biod{i}_y = []")

    count = 0
    for t in range(time_points):
        count += 1
        for i, biod in enumerate(shoal):  # this loop return an index i for each element biod in shoal
            if experimental_group == 'flock':
                print(experimental_group)
                biod.flock(shoal)
            elif experimental_group == 'random1':
                if count%5 == 0:
                    biod.random(maxSpeed/2)
            elif experimental_group == 'random3':
                    biod.random2(maxSpeed/2)
            speed = biod.update()
            print(speed)
            biod.borders(x_max, y_max, limit=50)  # 50 or 150
            exec(f'biod{i}_x.append(biod.location[0])')
            exec(f'biod{i}_y.append(biod.location[1])')
        time.append(t)

    # I create a dataframe with time and x and y for all my objects
    time = np.array(time)
    df = pd.DataFrame(dict(t=time))

    for i in range(n_biods):
        df[f'biod{i}_x'] = np.array(eval(f'biod{i}_x'))/x_max
        df[f'biod{i}_y'] = np.array(eval(f'biod{i}_y'))/y_max


# dataframe for different experimental groups
# df_biod_(type of movement:flock,random1,random2)_(number:3,4,6)_(speed:4,8,12)
# es. df_biod_flock_3_speed4

path = r'D:\Users\matilde.perrino\Documents\zebra\biodi_experiment\stimuli\\'

df.to_csv(path + f'df_{experimental_group}_{n_biods}_speed{(maxSpeed/2)}.csv')
