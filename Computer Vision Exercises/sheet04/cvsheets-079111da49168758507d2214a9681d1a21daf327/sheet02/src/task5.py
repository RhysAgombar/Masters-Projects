import cv2
import numpy as np

from task4 import get_intensity_gradient
inf = 25500000.0
def euc_distance1d(vector):

	k = 0
	v = [0]*(len(vector)+1) ## position of each root of parabolas
	z = [0.0]*(len(vector)+1)
	z[0] = -inf
	z[1] = inf
	##for each intersection point
	for q in range(1, len(vector)):
		intersec = ((vector[q] + q**2) - (vector[v[k]] + v[k]**2 )) / (2*q - 2*v[k])
		while intersec <= z[k]:##if intersection point is before the previous parabola intersection
			k -= 1
			intersec = ((vector[q] + q**2) - (vector[v[k]] + v[k]**2 )) / (2*q - 2*v[k])
		k+=1
		v[k] = q
		z[k] = intersec
		z[k+1] = inf

	k = 0
	distance = []
	for q in range(0, len(vector)):
		while z[k+1] < q:
			k += 1
		distance.append( (q-v[k])**2 + vector[v[k]])
	return distance


def euc_distance2d(edges):
	dmat = np.full(edges.shape, 1,dtype=np.float32)
	dmat = (dmat - edges/255)*inf # edge pixels will now be zero, the others 2550000
	##do 1d distance for each row

	for row in range(dmat.shape[0]):
		dmat[row] = euc_distance1d(dmat[row,:])
	for col in range(dmat.shape[1]):
		dmat[:,col] = euc_distance1d(dmat[:,col])

	return np.sqrt(dmat)

def dist_trans_cv(edges):
	dmat = np.full(edges.shape, 255,dtype=np.uint8)
	dmat = (dmat - edges).astype(np.uint8)# edge pixels will now be zero, the others 255
	return cv2.distanceTransform(dmat, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def task5():
	image = cv2.imread('data/traffic.jpg', 0)
	#gradient_direction, gradient_magnitude = get_intensity_gradient(image)
	canny = cv2.Canny(image,100,200)
	cv2.imshow("T5: canny edges",canny)

	## visualizing custom impl.
	eucD = euc_distance2d(canny)


	eucD_nor = ((eucD/eucD.max())*255.0  ).astype(np.uint8)

	eucD_heat = cv2.applyColorMap(eucD_nor, cv2.COLORMAP_JET)

	cv2.imshow("T5: own impl", eucD_heat)


	##visualizing cv2.distanceTransform

	cv_dist = dist_trans_cv(canny)

	cv_dist = ((cv_dist/cv_dist.max())*255.0  ).astype(np.uint8)
	cv_dist_heat = cv2.applyColorMap(cv_dist, cv2.COLORMAP_JET)
	cv2.imshow("T5: dist_trans_cv", cv_dist_heat )


	##calculate mean absolute difference
	diff = np.average(np.absolute(cv_dist - eucD_nor).astype(np.float32))
	print("abolute difference is %f" % diff)
if __name__ == '__main__':
	task5()
	print("Press any key to continue execution")
	cv2.waitKey(0)
