# terrain classes from visual terrain classification
# 0 = free
# 1 = rough
# 2 = obstacle
# 3 = stairs
# 4 = narrow section

Header header

# Size in cells
uint32 num_cells_x
uint32 num_cells_y

# Cell coordinates of (0,0)
uint32 origin_x
uint32 origin_y

# Resolution (m / cell)
float32 resolution

# Height values per cell (row step: cells_x)
uint32[] terrain_class
 
