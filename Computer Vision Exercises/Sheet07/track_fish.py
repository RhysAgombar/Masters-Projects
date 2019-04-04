import cv2
import numpy as np
import os


#from matplotlib import pyplot as plt

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

INIT_X = 448
INIT_Y = 191
INIT_WIDTH = 38
INIT_HEIGHT = 33

INIT_BBOX = [INIT_X, INIT_Y, INIT_WIDTH, INIT_HEIGHT]

def load_frame(frame_number):
    """
    :param frame_number: which frame number, [1, 32]
    :return: the image
    """
    image = cv2.imread(os.path.join(IMAGES_FOLDER, '%02d.png' % frame_number))
    return image


def crop_image(image, bbox):
    """
    crops an image to the bounding box
    """
    x, y, w, h = tuple(bbox)
    return image[y: y + h, x: x + w]


def draw_bbox(image, bbox, thickness=2, no_copy=False):
    """
    (optionally) makes a copy of the image and draws the bbox as a black rectangle.
    """
    x, y, w, h = tuple(bbox)
    if not no_copy:
        image = image.copy()
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness)

    return image


def compute_histogram(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    return(hist)

def compare_histogram(hist1, hist2):
    hist_comp_val = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)# cv2.HISTCMP_BHATTACHARYYA)

    likelihood = np.exp(-hist_comp_val )#/ 10.0**2)
    return likelihood

class Position(object):
    """
    A general class to represent position of tracked object.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_bbox(self):
        """
        since the width and height are fixed, we can do such a thing.
        """
        return [self.x, self.y, INIT_WIDTH, INIT_HEIGHT]

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Position(self.x * other, self.y * other)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "[%d %d]" % (self.x, self.y)

    def make_ready(self, image_width, image_height):
        # convert to int
        self.x = int(round(self.x))
        self.y = int(round(self.y))

        # make sure inside the frame
        self.x = max(self.x, 0)
        self.y = max(self.y, 0)
        self.x = min(self.x, image_width)
        self.y = min(self.y, image_height)


class ParticleFilter(object):
    def __init__(self, du, sigma, num_particles=200): #200
        self.template = None  # the template (histogram) of the object that is being tracked.
        self.position = None  # we don't know the initial position still!
        self.particles = []  # we will store list of particles at each step here for displaying.
        self.fitness = []  # particle's fitness values
        self.du = du
        self.sigma = sigma
        self.num_particles = num_particles


        self.mu = Position(0,0)

    def init(self, frame, bbox):
        self.position = Position(x=bbox[0], y=bbox[1])  # initializing the position

        for i in range(0,self.num_particles):
            self.particles.append(Position(int(np.random.normal(self.position.x, self.sigma, 1)), int(np.random.normal(self.position.y, self.sigma, 1))))

        fsum = 0
        for i in self.particles:
            bx = i.get_bbox()
            pimg = frame[bx[1]:bx[1]+bx[3],bx[0]:bx[0]+bx[2],:]
            phist = compute_histogram(pimg)

            cx = self.position.get_bbox()
            cimg = frame[cx[1]:cx[1]+cx[3],cx[0]:cx[0]+cx[2],:]
            chist = compute_histogram(cimg)
            dist = compare_histogram(phist, chist)
            self.fitness.append(dist)
            fsum += dist

        for i in range(len(self.fitness)):
            self.fitness[i] = self.fitness[i]/fsum

        #print(self.particles)
        return(0)

    def track(self, new_frame):


        #hld = self.fitness.index(max(self.fitness)) #self.num_particles[]
        #self.position = Position(x=self.particles[hld].x,y=self.particles[hld].y)



        ## Resample according to weights
        new_part = []
        new_part = np.random.choice(self.particles, self.num_particles, replace=True, p=self.fitness)
        self.particles = new_part.copy()

        mu = self.mu
        mn = Position(0,0)
        mno = Position(0,0)

        print("particles were:")
        st = ''
        for p in self.particles:
            st += str(p) + ";"
        print(st)
        print("mu is", mu.x, mu.y)

        #move particles
        for i in range(len(self.particles)):
            mn = mn + self.particles[i]
            self.particles[i] = Position(x=self.particles[i].x + int(np.random.normal(mu.x, self.sigma)),y=self.particles[i].y + int(np.random.normal(mu.y, self.sigma)))
            mno = mno + self.particles[i]

        print("particles are:")
        st = ''
        for p in self.particles:
            st += str(p) + ";"
        print(st)
        print("--------")

        #add some slight random noise to the sampling
        for i in range(len(self.particles)):
            self.particles[i].x = self.particles[i].x + int(np.random.normal(0, 5))
            self.particles[i].y = self.particles[i].y + int(np.random.normal(0, 5))

        #print(self.particles)

        ## Update fitness Functions for new particles
        fsum = 0
        self.fitness = []
        for i in self.particles:
            bx = i.get_bbox()
            if (bx[1] + bx[3] < new_frame.shape[0] and bx[0] + bx[2] < new_frame.shape[1]):
                pimg = new_frame[bx[1]:bx[1]+bx[3],bx[0]:bx[0]+bx[2],:]
                phist = compute_histogram(pimg)

                cx = self.position.get_bbox()
                cimg = new_frame[cx[1]:cx[1]+cx[3],cx[0]:cx[0]+cx[2],:]
                chist = compute_histogram(cimg)

                dist = compare_histogram(phist, chist)
                self.fitness.append(dist)
                fsum += dist
            else:
                self.fitness.append(0.0)

        for i in range(len(self.fitness)):
            self.fitness[i] = self.fitness[i]/fsum

        #update motion model
        mn.x = mn.x / len(self.particles)
        mn.y = mn.y / len(self.particles)
        mno.x = mno.x / len(self.particles)
        mno.y = mno.y / len(self.particles)

        self.mu.x = self.du * self.mu.x + (1 - self.du) * (mno.x - mn.x)
        self.mu.y = self.du * self.mu.y +(1 - self.du) * (mno.y - mn.y)

        return(0)

    def display(self, current_frame):
        cv2.imshow('frame', current_frame)

        frame_copy = current_frame.copy()
        for i in range(len(self.particles)):
            bx = self.particles[i].get_bbox()
            if (bx[1] + bx[3] < current_frame.shape[0] and bx[0] + bx[2] < current_frame.shape[1]):
                draw_bbox(frame_copy, self.particles[i].get_bbox(), thickness=1, no_copy=True)

        cv2.imshow('particles', frame_copy)
        cv2.waitKey(0)


def main():
    np.random.seed(0)
    DU = 10#?
    SIGMA = 10 #?

    cv2.namedWindow('particles')
    cv2.namedWindow('frame')
    frame_number = 1
    frame = load_frame(frame_number)

    tracker = ParticleFilter(du=DU, sigma=SIGMA,num_particles=200)
    tracker.init(frame, INIT_BBOX)
    tracker.display(frame)


    for frame_number in range(2, 33):
        frame = load_frame(frame_number)
        tracker.track(frame)
        tracker.display(frame)


if __name__ == "__main__":
    main()
