import cv2
import numpy as np
import maxflow

def question_3(I,rho=0.6,pairwise_cost_same=0.01,pairwise_cost_diff=0.5):

    ZERO = 0
    ONE = 255

    bONE = rho
    bZERO = 1-rho

    g = maxflow.Graph[float]()

    x,y = I.shape
    ids = g.add_grid_nodes((x,y))

    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):

            if(I[i][j] == ZERO):
                g.add_tedge(ids[i][j], (-1)*np.log(bZERO), (-1)*np.log(bONE))
            else:
                g.add_tedge(ids[i][j], (-1)*np.log(bONE), (-1)*np.log(bZERO))

            if(j < ids.shape[1]-1):
                if (I[i][j] == I[i][j+1]): #if labels are the same...
                    g.add_edge(ids[i][j], ids[i][j+1], pairwise_cost_same, pairwise_cost_same)
                else: #if labels are different...
                    g.add_edge(ids[i][j], ids[i][j+1], pairwise_cost_diff, pairwise_cost_diff)

            if(i < ids.shape[0]-1):
                if (I[i][j] == I[i+1][j]):
                    g.add_edge(ids[i][j], ids[i+1][j], pairwise_cost_same, pairwise_cost_same)
                else:
                    g.add_edge(ids[i][j], ids[i+1][j], pairwise_cost_diff, pairwise_cost_diff)

    g.maxflow()
    Denoised_I = np.zeros((x,y))
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            Denoised_I[i,j] = g.get_segment(ids[i][j])*255

    cv2.imshow('Original Img', I)
    cv2.imshow('Denoised Img', np.array(Denoised_I, dtype = np.uint8 )), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I,rho=0.6):

    labels = np.unique(I).tolist()

    Denoised_I = np.copy(I)
    ### Use Alpha expansion binary image for each label

    labels = [0, 128, 255]

    for l in range(len(labels)):
        g = maxflow.Graph[float]()
        x,y = I.shape
        ids = g.add_grid_nodes((x,y))

        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                if(I[i][j] == labels[l]):
                    g.add_tedge(ids[i][j], rho, (1-rho)/2)
                else:
                    g.add_tedge(ids[i][j], (1-rho)/2, rho)

                if(j < ids.shape[1]-1):

                    d = 1 # Potts cost
                    if (I[i][j] != I[i][j+1]):
                        d = 0
                    pairwise_cost = (1-d)

                    #pairwise_cost = (I[i][j] - I[i][j+1])**2
                    g.add_edge(ids[i][j], ids[i][j+1], pairwise_cost, pairwise_cost)

                if(i < ids.shape[0]-1):

                    d = 1 # Potts cost
                    if (I[i][j] != I[i+1][j]):
                        d = 0
                    pairwise_cost = (1-d)

                    #pairwise_cost = (I[i][j] - I[i+1][j])**2
                    g.add_edge(ids[i][j], ids[i+1][j], pairwise_cost, pairwise_cost)


        g.maxflow()
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                s = g.get_segment(ids[i][j])
                if (s == 0):
                    Denoised_I[i,j] = labels[l]

    cv2.imshow('Original Img', I)
    print(Denoised_I)
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.15)
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.3)
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.6)

    ### Call solution for question 4
    question_4(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()
