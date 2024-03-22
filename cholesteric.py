import numpy as np
from numpy import cos, sin, tan, arctan2, sqrt

'''Modelling of cholesteric liquid crystals. From Pyllama (GPL v3) (https://doi.org/10.1016/j.cpc.2021.108256) 
(needed to maintain consistent environment)
Plotting functions removed '''


# vec_u = unit vector
x_u = np.array([1, 0, 0])
y_u = np.array([0, 1, 0])
z_u = np.array([0, 0, 1])

class Cholesteric(object):
    """
    This class represents the directors of a cholesteric liquid crystal with a multilayer stack of rotating nematic
    layers. The physical model can be found in Frka-Petesic et al, Physical Review Materials, 2019, doi:10.1103/PhysRevMaterials.3.045601

    :param int pitch360: the (full) pitch for a 360 degree rotation in nanometers
    :param float tilt_rad: the tilt of the helical axis in radians
    :param int resolution: the number of planes over a 360 degree rotation
    :param int handedness: the value of ``handedness`` is ``+1`` for right-handed and ``-1`` for left-handed
    :param float n_exit: the refractive index of the stack’s exit isotropic semi-infinite medium
    :param int N_hel360: the number of full pitches to stack
    """

    def __init__(self, pitch360=500, tilt_rad=0, resolution=40, handedness=1, N_hel360=1):
        self.resolution = resolution
        self.handedness = handedness
        self.N_hel = N_hel360
        self.pitch = pitch360  # length in nm for a 360deg rotation
        self.q = handedness * 2 * np.pi / self.pitch
        self.tilt = tilt_rad  # tilt, angle in degrees between 0 and 90
        self.slicing = np.linspace(0, self.N_hel * pitch360, self.N_hel * self.resolution, endpoint=False)
        self.e1_u = np.array([cos(self.tilt), 0, -sin(self.tilt)])
        self.e2_u = np.array([0, 1, 0])
        self.e3_u = np.array([sin(self.tilt), 0, cos(self.tilt)])
        self.helical_axis = np.array([sin(self.tilt), 0, cos(self.tilt)])  # helical axis
        self.slices_rotangles = self.q * self.slicing
        self.slices_directors = [cos(self.tilt)*cos(iphi)*x_u + sin(iphi)*y_u -sin(self.tilt)*cos(iphi)*z_u for iphi in self.slices_rotangles]
        self.compression = 1
        self.history = []

    def copy(self):
        ch = Cholesteric()
        ch.resolution = self.resolution
        ch.pitch = self.pitch
        ch.q = self.q
        ch.tilt = self.tilt
        ch.slicing = self.slicing
        #ch.e1_u = self.e1_u
        #ch.e2_u = self.e2_u
        #ch.e3_u = self.e3_u
        ch.helical_axis = self.helical_axis
        ch.slices_rotangles = self.slices_rotangles
        ch.slices_directors = self.slices_directors
        return ch

    def compress(self, alpha):
        """
        This function updates the fields of the ``Cholesteric`` after vertical compression. The physical model can be
        found in Frka-Petesic et al, Physical Review Materials, 2019, doi:10.1103/PhysRevMaterials.3.045601

        :param float alpha: the coefficient of vertical compression, between 0 and 1
        """
        e1_u_new = np.array([cos(self.tilt), 0, -alpha*sin(self.tilt)])
        e2_u_new = np.array([0, 1, 0])
        e3_u_new = np.array([sin(self.tilt), 0, alpha*cos(self.tilt)])
        m_u_new = np.array([alpha*sin(self.tilt), 0, cos(self.tilt)])  # beta before compression
        beta_new = arctan2(alpha*sin(self.tilt), cos(self.tilt))
        p_new = self.pitch * sqrt(sin(beta_new)**2 + alpha**2*cos(beta_new)**2)
        phi_new = arctan2(p_new*sin(self.slices_rotangles), cos(self.slices_rotangles)*alpha*self.pitch)
        n_u_new = [cos(beta_new)*cos(iphi_new)*x_u + sin(iphi_new)*y_u + -sin(beta_new)*cos(iphi_new)*z_u for iphi_new in phi_new]
        # Save everything
        self.pitch = p_new
        self.q = 2 * np.pi / p_new
        self.tilt = beta_new
        self.slicing = np.linspace(0, self.N_hel * p_new, self.N_hel * self.resolution, endpoint=False)
        self.helical_axis = m_u_new
        self.slices_rotangles = phi_new
        self.slices_directors = n_u_new
        self.history.append('self.compress(' + str(alpha) + ')')

    def change_pitch(self, p_new):
        self.slicing = np.linspace(0, self.N_hel * p_new, self.N_hel * self.resolution, endpoint=False)
        self.pitch = p_new
        self.history.append('self.change_pitch(' + str(p_new) + ')')

    def change_tilt(self, beta_new):
        m_u_new = np.array([sin(beta_new), 0, cos(beta_new)])
        n_u_new = [cos(beta_new) * cos(iphi) * x_u + sin(iphi) * y_u + -sin(beta_new) * cos(iphi) * z_u for
                   iphi in self.slices_rotangles]
        self.helical_axis = m_u_new
        self.slices_directors = n_u_new
        self.tilt = beta_new
        self.history.append('self.change_tilt(' + str(beta_new) + ')')




