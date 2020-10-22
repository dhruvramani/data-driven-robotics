import os 
import sys
import numpy as np
import torch

from demons_config import get_demons_args

if __name__ == "__main__":
    demon_config = get_demons_args()
    deg = demon_config.deg()
    if demon_config.collect_by == 'teleop':
        # TODO remove this
        # tasks = ['beat_the_buzz', 'block_pyramid', 'change_channel', 'change_clock', 'close_box', 'close_door', 'close_drawer', 'close_fridge', 'close_grill', 'close_jar', 'close_laptop_lid', 'close_microwave', 'empty_container', 'empty_dishwasher', 'get_ice_from_fridge', 'hang_frame_on_hanger', 'hannoi_square', 'hit_ball_with_queue', 'hockey', 'insert_usb_in_computer', 'lamp_off', 'lamp_on', 'light_bulb_in', 'light_bulb_out', 'meat_off_grill', 'meat_on_grill', 'move_hanger', 'open_box', 'open_door', 'open_drawer', 'open_fridge', 'open_grill', 'open_jar', 'open_microwave', 'open_oven', 'open_window', 'open_wine_bottle', 'phone_on_base', 'pick_and_lift', 'pick_up_cup', 'place_cups', 'place_hanger_on_rack', 'place_shape_in_shape_sorter', 'play_jenga', 'plug_charger_in_power_supply', 'pour_from_cup_to_cup', 'press_switch', 'push_button', 'push_buttons', 'put_books_on_bookshelf', 'put_bottle_in_fridge', 'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_knife_in_knife_block', 'put_knife_on_chopping_board', 'put_money_in_safe', 'put_plate_in_colored_dish_rack', 'put_rubbish_in_bin', 'put_shoes_in_box', 'put_toilet_roll_on_stand', 'put_tray_in_oven', 'put_umbrella_in_umbrella_stand', 'reach_and_drag', 'reach_target', 'remove_cups', 'scoop_with_spatula', 'screw_nail', 'set_the_table', 'setup_checkers', 'slide_block_to_target', 'slide_cabinet_open', 'slide_cabinet_open_and_place_cups', 'solve_puzzle', 'stack_blocks', 'stack_cups', 'stack_wine', 'straighten_rope', 'sweep_to_dustpan', 'take_cup_out_from_cabinet', 'take_frame_off_hanger', 'take_item_out_of_drawer', 'take_lid_off_saucepan', 'take_money_out_safe', 'take_off_weighing_scales', 'take_plate_off_colored_dish_rack', 'take_shoes_out_of_box', 'take_toilet_roll_off_stand', 'take_tray_out_of_oven', 'take_umbrella_out_of_umbrella_stand', 'take_usb_out_of_computer', 'toilet_seat_down', 'toilet_seat_up', 'turn_oven_on', 'turn_tap', 'tv_off', 'tv_on', 'unplug_charger', 'water_plants', 'weighing_scales', 'wipe_desk']
        # for i, task in enumerate(tasks):
        #     print(i, task)
        #     try :
        deg.teleoperate(demon_config) #, task)
            # except :
            #     print("Couldn't collect demos")
    elif demon_config.collect_by == 'random':
        deg.random_trajectory(demon_config)
    elif demon_config.collect_by == 'imitation':
        #NOTE : NOT TESTED
        import imitate_play
        
        if demon_config.train_imitation:
            imitate_play.train_imitation(demon_config)
        imitate_play.imitate_play()