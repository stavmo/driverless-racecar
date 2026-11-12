# One traffic cone. Colors: blue = left edge, yellow = right edge,
# orange = start/finish line.
uint8 BLUE=0
uint8 YELLOW=1
uint8 ORANGE=2

float64 x      # [m] in the frame given by the parent message's header.frame_id
float64 y      # [m]
uint8 color
