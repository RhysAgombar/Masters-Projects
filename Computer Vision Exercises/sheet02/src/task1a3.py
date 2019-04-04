import cv2
import numpy as np
import time


def get_convolution_using_fourier_transform(image, kernel):
	ftim = np.fft.fft2(image)

	ftkep = np.zeros(image.shape)
	mx = int(np.round(image.shape[0]/2))
	my = int(np.round(image.shape[1]/2))

	ftkep[mx - 3:mx + 4, my - 3:my+  4] = kernel[:,:] #Put kernel into center of padded matrix of zeros, size of image

	ftke = np.fft.fft2(ftkep)

	ft = np.roll(np.roll(np.round(np.fft.ifft2(np.multiply(ftim, ftke))).astype(np.uint8), mx, axis=0), my, axis=1) # multiply the two FFTs together, then do an inverse FFT. The roll/round is to rearrange the image, since it gets shifted by the FFTs
	return ft

def get_g_kernel(mid, dim, sigma): # Creates the gaussian kernel
	g2 = np.zeros((dim,dim))
	for i in range(0,dim):
		for j in range(0,dim):
			g2[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-1*((i-mid)**2 + (j-mid)**2)/(2*sigma**2))
	return g2

def task1():
	image = cv2.imread('data/einstein.jpeg', 0)

	kernel = get_g_kernel(3, 7, 1) #calculate kernel (mid, dim, sigma)

	conv_result = cv2.filter2D(image,-1,kernel) #calculate convolution of image and kernel
	fft_result = get_convolution_using_fourier_transform(image, kernel)

	#compare results
	result = np.mean(np.abs(conv_result - fft_result))

	print(result)

#----------------------------------------------------------------------------------------------

def build_gaussian_pyramid_opencv(image, num_levels):
	pyramid = []
	hold = np.copy(image)
	pyramid.append(hold)
	for i in range(0, num_levels):
		hold = cv2.pyrDown(hold)
		pyramid.append(hold)
	return pyramid

def build_gaussian_pyramid(image, num_levels):
	pyramid = []
	hold = np.copy(image)
	pyramid.append(hold)
	dim = 5
	for i in range(0, num_levels):
		nimg = cv2.GaussianBlur(hold, (dim,dim), 0)
		nimg = nimg[::2, ::2] #downsample by factor of two
		pyramid.append(nimg)
		hold = nimg
	return pyramid

def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
	locations = ()
	dKernel = np.ones((2,2), np.uint8)
	for i in range(len(pyramid_image)-1,-1, -1):
		pImage = np.zeros((pyramid_image[i].shape[0] + pyramid_template[i].shape[0], pyramid_image[i].shape[1] + pyramid_template[i].shape[1]))
		pImage[0:pyramid_image[i].shape[0], 0:pyramid_image[i].shape[1]] = pyramid_image[i][:,:]
		out = np.zeros(pyramid_image[i].shape)

		if not locations:
			out = normalized_cross_correlation(pyramid_image[i],pyramid_template[i]) # Use NCC for lowest level of pyramid. It's so small at this state, that it takes almost no time
			out = cv2.dilate(out, dKernel, iterations = 1)
			locations = np.where(out > 250)
		else:
			tps = (pyramid_template[i] - np.mean(pyramid_template[i]))
			down = pyramid_template[i].shape[0]
			right = pyramid_template[i].shape[1]

			for y,x in zip(locations[0],locations[1]):
				y = y * 2
				x = x * 2

				imgs = (pImage[y:y+down, x:x+right] - np.mean(pImage[y:y+down, x:x+right])) # Top portion of the NCC formula
				sm = np.sum(tps*imgs)

				tps2 = np.sum(np.power(tps,2)) # Bottom portion of the NCC formula
				imgs2 = np.sum(np.power(imgs,2))
				sqrt = (tps2 * imgs2) ** 0.5

				out[y, x] = sm/sqrt

		out[np.where(out > 0.7)] = 255
		out[np.where(out <= 0.7)] = 0
		out = cv2.dilate(out, dKernel, iterations = 1)
		locations = np.where(out > 250)
		down = pyramid_template[i].shape[0]
		right = pyramid_template[i].shape[1]

		for k in range(0, len(locations[0])):
			cv2.rectangle(pyramid_image[i],(locations[1][k], locations[0][k]), (locations[1][k]+down, locations[0][k]+right), (255,0,0))

	return pyramid_image[0]

def normalized_cross_correlation(image, template):
	h = np.zeros(image.shape)
	mt = np.mean(template)

	pImage = np.zeros((image.shape[0] + template.shape[0],image.shape[1] + template.shape[1])) # Have to pad the image with 0s such that the bottom rows and right columns of the image can be analysed properly
	pImage[0:image.shape[0],0:image.shape[1]] = image[:,:]

	out = np.zeros(image.shape)
	for m in range(0, pImage.shape[0]-template.shape[0]):
		if (m %10 == 0):
			print("Progress: " + str(np.round((m/(pImage.shape[0]-template.shape[0]))*100, 1)) + "%")
		for n in range(0, pImage.shape[1]-template.shape[1]):
			tps = (template - np.mean(template)) # Top portion of the NCC formula
			imgs = (pImage[m:m+template.shape[0], n:n+template.shape[1]] - np.mean(pImage[m:m+template.shape[0], n:n+template.shape[1]]))
			sm = np.sum(tps*imgs)

			tps2 = np.sum(np.power(tps,2)) # Bottom portion of the NCC formula
			imgs2 = np.sum(np.power(imgs,2))
			sqrt = (tps2 * imgs2) ** 0.5
			out[m, n] = sm/sqrt

	out[np.where(out > 0.7)] = 255 # set matching pixels (with a confidence of > 0.7) to pure white

	return out


def task3():
	image = cv2.imread('data/traffic.jpg', 0)
	template = cv2.imread('data/traffic-template.png', 0)
	cv2.imshow("Traffic-Image", image)
	cv2.imshow("Traffic-Template", template)

	cv_pyramid = build_gaussian_pyramid_opencv(image, 4)
	mine_pyramid = build_gaussian_pyramid(image, 4)

	#compare and print mean absolute difference at each level
	print("Mean Differences for each level: ")
	for i in range(0, len(cv_pyramid)):
		print(np.mean(np.abs(cv_pyramid[i] - mine_pyramid[i])))
		# No difference when my blurring uses a kernel of size 5, however if I increase or decrease it, differences start to appear

	print("Beginning NCC Template Matching... This will take a while.")

	start = time.process_time()
	tmS = normalized_cross_correlation(image, template).astype(np.uint8)

	locations = np.where(tmS > 250) # Since we sent
	down = template.shape[0] # height of rectangle
	right = template.shape[1] # width of rectangle
	res1 = np.copy(image)
	for k in range(0, len(locations[0])):
		#Draw rectangles of size template at locations where the template matches
		cv2.rectangle(res1,(locations[1][k], locations[0][k]), (locations[1][k]+down, locations[0][k]+right), (255,0,0))
	end = time.process_time()

	print("Template Matching with NCC Time: " + str(end-start))
	print("Press any key to continue execution")
	cv2.imshow("Result - NCC", res1)
	cv2.waitKey(0)

	pyramid_template = build_gaussian_pyramid(template, 4)

	start = time.process_time()
	result = template_matching_multiple_scales(cv_pyramid, pyramid_template, 0.7)
	end = time.process_time()
	print("Template Matching with Pyramid Time: " + str(end-start))
	cv2.imshow("Result - Pyramid", result)
	cv2.waitKey(0)
if __name__ == '__main__':
	task1()
	task3()
	cv2.waitKey(0)
