import dynpssimpy.dynamic as dps
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt
import json
from sparculing.helper_functions import *
from sparculing.gen_sens import *


sys = dps.PowerSystemModel('k2a.json')

# pl0 = get_load_power_vector(model)
pl0 = sys.loads['Load'].par['P'].copy()

# pg0=get_gen_power_vector(model)
pg0 = sys.gen['GEN'].par['P'].copy()

sys.init_dyn_sim()
sys.gen['GEN'].idx

print(f"Before load change:\tP_load={sys.loads['Load'].par['P']}\tP_gen={sys.gen['GEN'].P_e(sys.x_0, sys.v_0)}")

# sys=dps.PowerSystemModel('k2a.json')
change_all_load_powers(sys, pl0+[-100,100])


           
# sys.setup_ready = False
sys.power_flow()
sys.init_dyn_sim()

print(f"After load change:\tP_load={sys.loads['Load'].par['P']}\tP_gen={sys.gen['GEN'].P_e(sys.x_0, sys.v_0)}")
# print(sys.s_0)