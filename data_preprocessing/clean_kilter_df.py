import json
import numpy as np

all_data = json.load(open('all_holds.json'))

main_hold_to_grid = {}
main_hold_to_coords = {}

# 18 by 17
mainhold_offset = 0
for idx, hold_idx in enumerate(all_data[0][0]['mainholds']['hold_id']):
    main_hold_to_grid[hold_idx] = all_data[0][0]['mainholds']['grid'][idx]
    main_hold_to_coords[hold_idx] = all_data[0][0]['mainholds']['coords'][idx]

# 15 by 18
aux_hold_offset = 18
aux_hold_to_grid = {}
aux_hold_to_coords = {}
for idx, hold_idx in enumerate(all_data[0][1]['auxillary']['hold_id']):
    aux_hold_to_grid[hold_idx] = all_data[0][1]['auxillary']['grid'][idx]
    aux_hold_to_coords[hold_idx] = all_data[0][1]['auxillary']['coords'][idx]

# 2 by 18
kickboard_offset = 33
kickboard_to_grid = {}
kickboard_to_coords = {}
for idx, hold_idx in enumerate(all_data[0][2]['kickboard']['hold_id']):
    kickboard_to_grid[hold_idx] = all_data[0][2]['kickboard']['grid'][idx]
    kickboard_to_coords[hold_idx] = all_data[0][2]['kickboard']['coords'][idx]

# convert a list of holds and hold types into a one hot encoding
def convert_hold_to_one_hot(hold_idxs, channels):
    final_grid = np.zeros((4, 35, 18))
    for channel, hold_idx in zip(channels, hold_idxs):
        # classify if this hold is a main hold, auxillary hold, or kick hold
        if hold_idx in main_hold_to_grid:
            main_j, main_i = main_hold_to_grid[hold_idx]
            if main_i > 2:
                main_i += (main_i - 2)
            final_grid[channel, main_i, main_j] = 1

        elif hold_idx in aux_hold_to_grid:
            aux_j, aux_i = aux_hold_to_grid[hold_idx]
            aux_i += aux_i +3
            aux_j = aux_j * 2
            final_grid[channel, aux_i, aux_j] = 1

        elif hold_idx in kickboard_to_grid:
            kick_j, kick_i = kickboard_to_grid[hold_idx]
            final_grid[channel, kick_i+kickboard_offset, kick_j] = 1

    assert(np.sum(final_grid) > 1)
    return final_grid

def htype_to_channel(htype):
    if htype == '12':
        return 0
    elif htype == '13':
        return 1
    elif htype == '14':
        return 2
    else:
        return 3

csv = open('kilter_df.csv')
next(csv)

climbs = []
vgrades = []
ascents = []
qualities = []
angles = []

for line in csv:
    toks = line.strip().split(',')[1:]
    # ['uuid', 'board_angle', 'frames', 'quality_average', 'v_grade', 'ascensionist_count']
    uuid, board_angle, frames, quality_average, vgrade, ascensionist_count = toks
    frames = frames.split('p')[1:]

    # holds and hold types (converted into channel)
    holds = [int(hold_and_type.split('r')[0]) for hold_and_type in frames]
    hold_types = [htype_to_channel(hold_and_type.split('r')[1]) for hold_and_type in frames]

    curr_climb_one_hot = convert_hold_to_one_hot(holds, hold_types)

    climbs.append(curr_climb_one_hot)
    vgrades.append(float(vgrade))
    ascents.append(float(ascensionist_count))
    qualities.append(float(quality_average))
    angles.append(float(board_angle))

climbs = np.array(climbs)
vgrades = np.array(vgrades)
ascents = np.array(ascents)
qualities = np.array(qualities)
angles = np.array(angles)

np.save('../training_data/kilter_climb_features.npy', climbs)
np.save('../training_data/kilter_vgrades.npy', vgrades)
np.save('../training_data/kilter_ascents.npy', ascents)
np.save('../training_data/kilter_qualities.npy', qualities)
np.save('../training_data/kilter_angles.npy', angles)
