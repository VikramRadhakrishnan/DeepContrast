import tensorflow as tf

def fraunhofer_tf(input_grid, output_grid, wavelength, field):
  uv_grid = output_grid.scaled(2 * np.pi/ wavelength)
  M1 = tf.convert_to_tensor(np.exp(-1j * np.outer(uv_grid.separated_coords[1], input_grid.separated_coords[1])))
  M2 = tf.convert_to_tensor(np.exp(-1j * np.outer(uv_grid.separated_coords[0], input_grid.separated_coords[0])).T)
  
  f = tf.reshape(field * input_grid.weights, (M1.shape[1], M2.shape[0]))
  result = -1j * tf.matmul(M1, tf.matmul(f, M2)) / wavelength
  return tf.reshape(result, [-1])

def contrast_metric(x, D, coron, ncp_field, dh_mask, cent_ind, E_in, wavelength):
  '''Function to calculate the gradient of the contrast w.r.t. the DM actuators.
  x: Tensor variable with actuator values of the DM.
  D: Influence functions of the DM.
  coron: Electric field contribution from the coronagraph optics.
  ncp: Electric field contribution from non common path aberrations.
  dh_ind: Dark hole mask in the focal plane where contrast is evaluated.
  cent_ind: Central bright pixel indices in the focal plane used to evaluate contrast.
  E_in: Placeholder for the incident electric field being propagated through the optics.
  wavelength: Wavelength of the incident wavefront'''
  
  # Use the DM influence functions and actuator levels to determine the surface
  inf_mat = tf.constant(D, name="inf_mat")
  S = tf.linalg.matvec(inf_mat, x, name="dm_surface")
  
  # Calculate the electric field contribution from the DM
  dm_field = tf.exp(2j * (2 * np.pi / wavelength) * tf.cast(S, tf.complex128), name="dm_field")
  E_d = tf.math.multiply(E_in, dm_field, name="E_d")
  
  # Calculate the electric field contribution from the NCPA
  E_ncp = tf.math.multiply(E_d, ncp_field, name="E_ncp")
  
  # Calculate the electric field contribution from the coronagraph
  coron_field = tf.constant(coron, dtype=tf.complex128, name="coron")
  E_c = tf.math.multiply(E_ncp, coron_field, name="E_c")
  
  # Fraunhofer propagate the electric field to the focal plane
  E_f = fraunhofer_tf(pupil_grid, science_focal_grid, wavelength, E_c)
  
  # Calculate the focal plane intensity
  I_f = tf.square(tf.abs(E_f), name="I_f")
  
  # Extract the intensities from pixels at the dark hole and at the center
  dh_intens = tf.divide(tf.math.reduce_sum(tf.math.multiply(I_f, dh_mask)), tf.math.reduce_sum(dh_mask))
  cent_intens = I_f[cent_ind]
  
  # Calculate the average raw contrast in the dark hole
  contrast = tf.math.divide(dh_intens, cent_intens, name='contrast')
                      
  return contrast
