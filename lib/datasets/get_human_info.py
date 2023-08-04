import os
import sys
from lib.config import cfg
import numpy as np
import re 

def get_human_info(split):

    if split == 'train':
        human_info = {
                'CoreView_313': {'begin_i': 0, 'i_intv': 1, 'ni': 60},
                'CoreView_315': {'begin_i': 0, 'i_intv': 6, 'ni': 400},
                'CoreView_377': {'begin_i': 0, 'i_intv': 30, 'ni': 300},
                'CoreView_386': {'begin_i': 0, 'i_intv': 6, 'ni': 300},
                'CoreView_390': {'begin_i': 700, 'i_intv': 6, 'ni': 300}, # 700 - 900
                'CoreView_392': {'begin_i': 0, 'i_intv': 6, 'ni': 300},
                'CoreView_396': {'begin_i': 810, 'i_intv': 5, 'ni': 270}
                }

    elif split == 'test':
        if cfg.test.mode == 'model_o_motion_o':
            # fitting 
            human_info = {
                'CoreView_313': {'begin_i': 0, 'i_intv': 1, 'ni': 60},
                'CoreView_315': {'begin_i': 0, 'i_intv': 1, 'ni': 400},
                'CoreView_377': {'begin_i': 0, 'i_intv': 1, 'ni': 300},
                'CoreView_386': {'begin_i': 0, 'i_intv':     1, 'ni': 300},
                'CoreView_390': {'begin_i': 700, 'i_intv': 1, 'ni': 300},
                'CoreView_392': {'begin_i': 0, 'i_intv': 1, 'ni': 300},
                'CoreView_396': {'begin_i': 810, 'i_intv': 1, 'ni': 270}
                }
        elif cfg.test.mode == 'model_o_motion_x':
            # pose generalization 
            human_info = {
                'CoreView_313': {'begin_i': 60, 'i_intv': 1, 'ni': 1000},
                'CoreView_315': {'begin_i': 400, 'i_intv': 1, 'ni': 1000},
                'CoreView_377': {'begin_i': 300, 'i_intv': 1, 'ni': 317},
                'CoreView_386': {'begin_i': 300, 'i_intv': 1, 'ni': 346},
                'CoreView_390': {'begin_i': 0, 'i_intv': 1, 'ni': 700},
                'CoreView_392': {'begin_i': 300, 'i_intv': 1, 'ni': 256},
                'CoreView_396': {'begin_i': 1080, 'i_intv': 1, 'ni': 270}
                }
        elif cfg.test.mode == 'model_x_motion_x':
            # identity generalization 
            human_info = {
                'CoreView_387': {'begin_i': 0, 'i_intv': 1, 'ni': 654}, 
                'CoreView_393': {'begin_i': 0, 'i_intv': 1, 'ni': 658},  
                'CoreView_394': {'begin_i': 0, 'i_intv': 1, 'ni': 859}  
                }
            
    return human_info

def get_human_info_h36m():

    human_info = {
        'S1': {'begin_i': 0, 'i_intv': 5, 'ni': 150}, # 0-300
        'S5': {'begin_i': 0, 'i_intv': 5, 'ni': 250},  # 0-300 
        'S6': {'begin_i': 0, 'i_intv': 5, 'ni': 150},  # 0-300
        'S7': {'begin_i': 0, 'i_intv': 5, 'ni': 300},  # 0-300
        'S8': {'begin_i': 0, 'i_intv': 5, 'ni': 250},  # 0-300
        'S9': {'begin_i': 0, 'i_intv': 5, 'ni': 260},  # 0-300
        'S11': {'begin_i': 0, 'i_intv': 5, 'ni': 200}  # 0-300
        }
    
    return human_info

def get_human_info_gpnerf(split):
    # copied from GP_NeRF official code. 
    human_info = {}
    if split == 'train':
        # CoreView_313
        CoreView_313 = {
            'begin_i': 1,
            'i_intv': 1,
            'ni': 300,
        }
        human_info['CoreView_313'] = CoreView_313
        
        # CoreView_315
        CoreView_315 = {
            'begin_i': 1,
            'i_intv': 1,
            'ni': 300,
        }
        human_info['CoreView_315'] = CoreView_315
        
        # CoreView_377
        CoreView_377 = {
            'begin_i': 0,
            'i_intv': 1,
            'ni': 300,
        }
        human_info['CoreView_377'] = CoreView_377
        
        # CoreView_386
        CoreView_386 = {
            'begin_i': 0,
            'i_intv': 1,
            'ni': 300,
        }
        human_info['CoreView_386'] = CoreView_386
        
        # CoreView_390 (lack vis map)
        CoreView_390 = {
            'begin_i': 700,
            'i_intv': 1,
            'ni': 300,
        }
        human_info['CoreView_390'] = CoreView_390
        
        # CoreView_394
        CoreView_394 = {
            'begin_i': 0,
            'i_intv': 1,
            'ni': 300,
        }
        human_info['CoreView_394'] = CoreView_394
        
        # CoreView_396
        CoreView_396 = {
            'begin_i': 810,
            'i_intv': 1,
            'ni': 300,
        }
        human_info['CoreView_396'] = CoreView_396
    
    elif split == 'test':
        if cfg.test.mode == 'model_x_motion_x':
            # CoreView_387 (Test)
            CoreView_387 = {
                'begin_i': 0,
                'i_intv': 1,
                'ni': 300,
            }
            human_info['CoreView_387'] = CoreView_387
            
            
            # CoreView_392 (Test)
            CoreView_392 = {
                'begin_i': 0,
                'i_intv': 1,
                'ni': 300,
            }
            human_info['CoreView_392'] = CoreView_392
            
            # CoreView_393 (Test)
            CoreView_393 = {
                'begin_i': 0,
                'i_intv': 1,
                'ni': 300,
            }
            human_info['CoreView_393'] = CoreView_393
        
    return human_info

def get_human_info_thu():
    ### 10 random human and the first half of the sequnce.
    human_info = {
                  'results_gyx_20181011_wlf_1_M': {'begin_i': 0, 'i_intv': 1, 'ni': 16}, 
                  'results_gyx_20181013_xyz_1_F': {'begin_i': 0, 'i_intv': 1, 'ni': 18}, 
                  'results_gyx_20181011_zxh_2_M': {'begin_i': 0, 'i_intv': 1, 'ni': 20}, 
                  'results_gyx_20181013_hj_1_F': {'begin_i': 0, 'i_intv': 1, 'ni': 19}, 
                  'results_gyx_20181015_hzx_1_M': {'begin_i': 0, 'i_intv': 1, 'ni': 10}, 
                  'results_gyx_20181011_scw_1_M': {'begin_i': 0, 'i_intv': 1, 'ni': 21}, 
                  'results_gyx_20181013_zy_2_M': {'begin_i': 0, 'i_intv': 1, 'ni': 13}, 
                  'results_gyx_20181015_dc_1_F': {'begin_i': 0, 'i_intv': 1, 'ni': 16}, 
                  'results_gyx_20181013_fk_1_M': {'begin_i': 0, 'i_intv': 1, 'ni': 16}, 
                  'results_gyx_20181011_lty_1_M': {'begin_i': 0, 'i_intv': 1, 'ni': 18}
                  }
    
    return human_info