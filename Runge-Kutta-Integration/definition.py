# import ctypes

# path = "D:/Documents/Coding/Repositorios/Meus Repositórios/Symbolic-Regression/Phase Three - SR/Runge-Kutta-Integration/clibrary.so"
# clibrary = ctypes.CDLL(path)

# clibrary.main()

import sys
sys.path.insert(1, r"D:\Program Files (x86)\TuringBot")

import turingbot as tb
import time

path = r'D:\Program Files (x86)\TuringBot\TuringBot.exe' 
input_file = r'D:\Documents\Coding\Repositorios\Meus Repositórios\Symbolic-Regression\Phase Three - SR\Runge-Kutta-Integration\rungekutta_prepared.csv' 
config_file = r'D:\Documents\Coding\Repositorios\Meus Repositórios\Symbolic-Regression\Phase Three - SR\Runge-Kutta-Integration\settings.cfg' 


# sim = tb.simulation()

def train_once(sleep_time):
    sim = tb.simulation()

    sim.start_process(path, input_file, threads=4, config="settings.cfg")

    time.sleep(sleep_time)

    sim.refresh_functions()

    to_return = sim.functions

    sim.terminate_process()

    return to_return

# functions = train_once(2)

# print(functions)

# print(*functions, sep='\n')

# sim.terminate_process()