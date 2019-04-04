import cv2
import numpy as np
def get_gaussian_kernel(x,y,sigma):
	gaussian = lambda x,y:  (1.0 / 2*np.pi*(sigma**2))*np.exp(- (x**2 + y**2)/(2*sigma**2))
	kernel = np.zeros((x,y),dtype=float)
	side_length_x = x/2
	side_length_y = y/2
	for i in range(x):
		for j in range(y):
			kernel[i][j] = gaussian((i-side_length_x), (j-side_length_y))
	return kernel

def get_intensity_gradient(image):
	#generate a larger gaussian kernel to cut borders when convoluting
	gaussian = get_gaussian_kernel(7,7,0.6)
	gaussian_devx = np.zeros((5,5))
	gaussian_devy = np.zeros((5,5))
	#calculate derivate x
	for y in range(0,gaussian_devx.shape[0]):##for each line
		for x in range(0,gaussian_devx.shape[1]):##for each column
			gaussian_devx[y][x] = gaussian[(y+1)][(x+1)] + -1*gaussian[(y+1)][(x+1)+1]
	#print("sum gaussian_devx: %f" % gaussian_devx.sum())


	#calculate derivate y
	for y in range(0,gaussian_devy.shape[0]):##for each line
		for x in range(0,gaussian_devy.shape[1]):##for each column
			gaussian_devy[y][x] = gaussian[(y+1)][(x+1)] + -1*gaussian[(y+1)+1][(x+1)]
	#print("sum gaussian_devy: %f" % gaussian_devy.sum())




	edges_x = cv2.filter2D(image, -1, gaussian_devx)
	cv2.imshow("T4: edges x", edges_x)
	edges_y = cv2.filter2D(image, -1, gaussian_devy)
	cv2.imshow("T4: edges y", edges_y)


	###calculating gradient
	gradient_direction = np.arctan2(edges_y, edges_x )
	gradient_magnitude = np.sqrt(edges_y*edges_y + edges_x*edges_x)
	return (gradient_direction, gradient_magnitude)

def task4():

	image = cv2.imread('data/einstein.jpeg', 0)
	gradient_direction, gradient_magnitude = get_intensity_gradient(image)
	#redistribute magnitude
	grad_mag_show = (gradient_magnitude*255.0)/gradient_magnitude.max()
	grad_mag_show = grad_mag_show.astype(np.uint8)
	#redistribute direction
	grad_direc_show = (gradient_direction*255.0)/gradient_direction.max()
	grad_direc_show = grad_direc_show.astype(np.uint8)

	cv2.imshow("T4: gradient magnitude",grad_mag_show)
	cv2.imshow("T4: gradient direction",grad_direc_show)



if __name__ == '__main__':
	task4()
	print("Press any key to continue execution")
	cv2.waitKey(0)
