import numpy as np
import time

import utils
import task1
import task2

hands_orig_train = 'data/hands_orig_train.txt.new'
hands_aligned_test = 'data/hands_aligned_test.txt.new'
hands_aligned_train = 'data/hands_aligned_train.txt.new'

'''
    returns a (#hands, #landmarks, #dim) matrix
'''
def get_keypoints(path):
    data_info = utils.load_data(path)


    return utils.convert_samples_to_xy(data_info['samples'])

def task_1():
    # Loading Trainig Data
    kpts = get_keypoints(hands_orig_train)

    # calculate mean
	##picking one data instance as the mean
    mean_data = kpts[0,:]


    # we want to visualize the data first

    utils.visualize_hands(kpts,"hands",delay=0.001)

    task1.procrustres_analysis(kpts)


def task_2_1():
    # ============= Load Data =================
    kpts = get_keypoints(hands_aligned_train)

    mean, pcs, pc_weights = task2.train_statistical_shape_model(kpts)

    return mean, pcs, pc_weights

def task_2_2(mean, pcs, pc_weights):

    kpts = get_keypoints(hands_aligned_test)

    reconst = task2.reconstruct_test_shape(kpts, mean, pcs, pc_weights, title="Test Reconstruction from Model")
    utils.visualize_hands(kpts, "Actual Test Points", delay=10, ax=None, clear=False)

    error = np.sqrt(np.sum(np.power(kpts - reconst,2)) / kpts.shape[0])
    print("RMS Error: ", error)

    time.sleep(20)


if __name__ == '__main__':
    print("Running Task 1")
    task_1()

    print("Running Task 2.1")
    mean, pcs, pc_weights = task_2_1()

    print("Running Task 2.2")
    task_2_2(mean, pcs, pc_weights)
