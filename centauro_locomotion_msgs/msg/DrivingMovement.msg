# This message describes the desired robot center driving movement in longitudinal direction (x), 
# lateral direction (y) and around the z_axis (omega). Leg configurations are fixed. 
# This message will arrive in a rate between 20 and 50 Hz. If some delay occurs due to
# steering wheels in the required direction, this is no problem.

Header header

float32 vel_x
float32 vel_y
float32 omega
