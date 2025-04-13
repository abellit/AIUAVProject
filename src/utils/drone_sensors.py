import numpy 


class Sensors:

    def __init__(self):
        self.accelerometer = numpy.array([0, 0, 0])
        self.gyroscope = numpy.array([0, 0, 0])
        self.magnetometer = numpy.array([0, 0, 0])
        self.altimeter = 0
        self.gps = numpy.array([0, 0, 0])
        self.barometer = 0
        self.rgb_camera = None
        self.depth_camera = None
        self.infrared_camera = None
        self.lidar = None
        self.imu = None
      


    def set_accelerometer(self, x, y, z):
        self.accelerometer = numpy.array([x, y, z])

    def set_gyroscope(self, x, y, z):
        self.gyroscope = numpy.array([x, y, z])

    def set_magnetometer(self, x, y, z):
        self.magnetometer = numpy.array([x, y, z])

    def set_altimeter(self, altimeter):
        self.altimeter = altimeter

    def set_gps(self, x, y, z):
        self.gps = numpy.array([x, y, z])

    def set_battery(self, battery):
        self.battery = battery

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_pressure(self, pressure):
        self.pressure = pressure

    def get_accelerometer(self):
        return self.accelerometer

    def get_gyroscope(self):
        return self.gyroscope

    def get_magnetometer(self):
        return self.magnetometer

    def get_altimeter(self):
        return self.altimeter

    def get_gps(self):
        return self.gps

    def get_battery(self):
        return self.battery

    def get_temperature(self):
        return self.temperature

    def get_pressure(self):
        return self.pressure