import time
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -0.42193

def timefn(fn):
	@wraps(fn)
	def measure_time(*args, **kwargs):
		t1 = time.time()
		result = fn(*args, **kwargs)
		t2 = time.time()
		print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
		return result
	return measure_time

def calc_pure_python(desired_width, max_iterations):
	"""Create a list of complex coordinates (zs) and complex parameters (cs),
	build Julia Set"""
	x_step = (x2 - x1) / desired_width
	y_step = (y1 - y2) / desired_width
	x = []
	y = []
	ycoord = y2
	while ycoord > y1:
		y.append(ycoord)
		ycoord += y_step
	xcoord = x1
	while xcoord < x2:
		x.append(xcoord)
		xcoord += x_step
	# build a list of coordinates and the initial condition for each cell.
	# Note that our initial condition is a constant and could easily be removed,
	# we use it to simulate a real-world scenario with several inputs to our
	# function
	zs = []
	cs = []
	for ycoord in y:
		for xcoord in x:
			zs.append(complex(xcoord, ycoord))
			cs.append(complex(c_real, c_imag))
	print("Length of x:", len(x))
	print("Total elements:", len(zs))
	start_time = time.time()
	output = calculate_z_serial_purepython(max_iterations, zs, cs)
	end_time = time.time()
	secs = end_time - start_time
	print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

	# This sum is expected for a 1000^2 grid with 300 iterations
	# It ensures that our code evolves exactly as we'd intended
	assert sum(output) == 33219980
	return zs, cs, output

#@timefn
def calculate_z_serial_purepython(maxiter, zs, cs):
	"""Calculate output list using Julia update rule"""
	output = [0] * len(zs)
	for i in range(len(zs)):
		n = 0
		z = zs[i]
		c = cs[i]
		while abs(z) < 2 and n < maxiter:
			z = z * z + c
			n += 1
		output[i] = n
	return output

def draw_false_grayscale(complex_list, iteration_list):
	real_list = [c.real for c in complex_list]
	imag_list = [c.imag for c in complex_list]
	min_real, max_real = min(real_list), max(real_list)
	min_imag, max_imag = min(imag_list), max(imag_list)
	hist, bins = np.histogram(iteration_list, bins=50, range=(np.min(iteration_list), np.max(iteration_list)), density=True)
	cdf = hist.cumsum()
	cdf = 255 * cdf / cdf[-1]
	iterations_equalized = np.interp(iteration_list, bins[:-1], cdf)
	iteration_list_2d = np.reshape(iterations_equalized, (1000, 1000))
	norm_iterations = iteration_list_2d / np.max(iteration_list_2d)
	plt.imshow(norm_iterations, cmap='gray', extent=(min_real, max_real, min_imag, max_imag))
	plt.colorbar(label='Iteration Count')
	plt.xlabel('Real part')
	plt.ylabel('Imaginary part')
	plt.title('Julia Set Visualization')
	plt.show()

def draw_pure_grayscale(complex_list, iteration_list):
	real_list = [c.real for c in complex_list]
	imag_list = [c.imag for c in complex_list]
	min_real, max_real = min(real_list), max(real_list)
	min_imag, max_imag = min(imag_list), max(imag_list)
	pure_list = [i == 300 for i in iteration_list]
	pure_list_2d = np.reshape(pure_list, (1000, 1000))
	plt.imshow(pure_list_2d, cmap='gray', extent=(min_real, max_real, min_imag, max_imag))
	plt.colorbar(label='Iteration Count')
	plt.xlabel('Real part')
	plt.ylabel('Imaginary part')
	plt.title('Julia Set Visualization')
	plt.show()

if __name__ == "__main__":
	# Calculate the Julia set using a pure Python solution with
	# reasonable defaults for a laptop
	zs, cs, output = calc_pure_python(desired_width=1000, max_iterations=300)
	# Uncomment to get the drawings of the Julia Set
	#draw_false_grayscale(zs, output)
	#draw_pure_grayscale(zs, output)

	