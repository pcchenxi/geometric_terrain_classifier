Header header

# Size in cells
uint32 cells_x
uint32 cells_y

# Cell coordinates of (0,0)
uint32 origin_x
uint32 origin_y

# Resolution (m / cell)
float32 resolution

# Height values per cell (row step: cells_x)
float32[] height
