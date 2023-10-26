def control(piracer=None, direction=2):
    if piracer is None:
        return

    if direction == 0: scaled_percent = -1    # a little left
    if direction == 1: scaled_percent = -1.5  # left
    if direction == 2: scaled_percent = 0     # straight
    if direction == 3: scaled_percent = 1    # a little right
    if direction == 4: scaled_percent = 1.5     # right

    piracer.set_steering_percent(scaled_percent)
