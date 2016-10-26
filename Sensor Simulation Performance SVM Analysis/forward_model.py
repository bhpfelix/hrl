
# attempt at modeling temperature flow thorugh a material
import numpy as np
import matplotlib.pyplot as pp
import math

class model_temperature:

	def __init__(self, t_sens_0, t_amb, k_sens, alpha_sens, k_obj, alpha_obj):
		self.t_ambient = t_amb + 273.15 # in Kelvin : 25 celsius
		self.t_sens_0 = t_sens_0 + 273.15
		self.t_obj_0 = self.t_ambient

		self.sampling_time = 0.005 # in seconds
		self.x = 0.08*0.001 # m : depth of thermistor from surface
		
		self.k_sens = k_sens
		self.alpha_sens = alpha_sens
		self.k_obj = k_obj
		self.alpha_obj = alpha_obj

		self.temp_list = []        

	def run_simulation(self):
		t_surf = (self.t_sens_0*(self.k_sens/math.sqrt(self.alpha_sens)) + self.t_obj_0*(self.k_obj/math.sqrt(self.alpha_obj)))/(self.k_sens/math.sqrt(self.alpha_sens) + self.k_obj/math.sqrt(self.alpha_obj))
		time_list = np.arange(0.01,10,self.sampling_time)
		for ts in time_list:
			self.temp_list.append(self.t_sens_0 + (t_surf - self.t_sens_0)*math.erfc(self.x/(2*math.sqrt(self.alpha_sens*ts))))
		self.temp_list = self.temp_list + np.random.normal(0,0.001,len(self.temp_list))
		return time_list, self.temp_list

	def visualize_temp(self, time, data):
		pp.figure()
		pp.title('Heat Flow',fontsize='24')
		pp.xlabel('Time (s)',fontsize='24')
		pp.ylabel('Temperature (K)',fontsize='24')
		pp.plot(time, data, linewidth=4.0, color='g')
		pp.hold(True)
		pp.xlim([0,10])
		pp.grid('on')

if __name__ == '__main__':
    
	#from identify_sensor_parameters import k_sens, alpha_sens
	k_sens = 0.0349
	alpha_sens = 2.796*10**(-9)
	t_sens_0 = 30
	t_amb = 25
	k_obj = 0.15
	alpha_obj = 0.15/(440.*1660.)
		
	temp_models = model_temperature(t_sens_0, t_amb, k_sens, alpha_sens, k_obj, alpha_obj)
	time, temp_values = temp_models.run_simulation()
	temp_models.visualize_temp(time, temp_values)

	pp.show()

