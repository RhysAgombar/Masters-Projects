import cv2
import numpy as np

def normalized_cross_correlation(image, template):
	h = np.zeros(image.shape)
	mt = np.mean(template)
	template_mean = np.mean(template)
	pImage = np.zeros((image.shape[0] + template.shape[0],image.shape[1] + template.shape[1])) # Have to pad the image with 0s such that the bottom rows and right columns of the image can be analysed properly
	pImage[0:image.shape[0],0:image.shape[1]] = image[:,:]

	out = np.zeros(image.shape)

	tps = (template - template_mean) # Top portion of the NCC formula
	tps2 = np.sum(np.power(tps,2)) # Bottom portion of the NCC formula
	for m in range(0, pImage.shape[0]-template.shape[0]):
		for n in range(0, pImage.shape[1]-template.shape[1]):
			imgs = (pImage[m:m+template.shape[0], n:n+template.shape[1]] - np.mean(pImage[m:m+template.shape[0], n:n+template.shape[1]]))
			sm = np.sum(tps*imgs)
			imgs2 = np.sum(np.power(imgs,2))
			sqrt = (tps2 * imgs2) ** 0.5
			cor = sm/sqrt
			if cor >= 0.7:
				out[m,n]= 255

	return out

def template_matching(image, template):
	matches = normalized_cross_correlation(image,template)

	print("matches shape is "+ str(np.where(matches == 255)))
	template_x, template_y = template.shape
	x_l, y_l = np.where(matches == 255)
	output = np.copy(image)
	for x,y in zip(x_l, y_l):
		xlen = template.shape[1]
		ylen = template.shape[0]
		output = cv2.rectangle(output,(x,y), (x+xlen, y+ylen), 255) # Draw rectangle using top left and bottom right points, and generate random colour

	return output
def task2():
	image = cv2.imread('data/lena.png', 0)
	template = cv2.imread('data/eye.png', 0)
	cv2.imshow("T2: image",image)
	cv2.imshow("T2: template",template)
	out = template_matching(image, template)
	cv2.imshow("T2: out",out)


	#mask = (image*(out/255)).astype("uint8")
	#cv2.imshow("image masked", mask)
if __name__ == '__main__':
	task2()
	print("Press any key to continue execution")
	cv2.waitKey(0)
