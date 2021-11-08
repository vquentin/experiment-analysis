import utils.masks as masks

def distance_to_collector(relative_d, mask_name):
    d1 = masks.masks[mask_name]['distance_first_cavity']
    d2 = masks.masks[mask_name]['distance_center_collector']
    return abs(abs((relative_d+d1)-(d2))-d2)
