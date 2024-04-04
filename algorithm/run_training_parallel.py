import time
from Training_differentiated import *
from Training_generations_increased import *
import concurrent.futures
import os

if __name__ == "__main__":
    populations = [20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200,
               250, 300, 350, 400, 550, 600, 750, 800, 950, 1000, 
               2000, 3000, 4000, 5000, 6000, 7000]

    DIR_PATH = "output_differentiated/"

    if os.path.isfile(DIR_PATH + "results.csv"):
        raise OSError("File exists")
    else:
        with open(DIR_PATH + "results.csv", "w") as file:
            file.write("population_size, fitness_score, training_time\n")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(train_inten_diff, populations)
    
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    DIR_PATH = "output_diff_scaled/"

    if os.path.isfile(DIR_PATH + "results.csv"):
        raise OSError("File exists")
    else:
        with open(DIR_PATH + "results.csv", "w") as file:
            file.write("population_size, fitness_score, training_time\n")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(train_iten_diff_scaled, populations)

    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    generations = [i for i in range(29, 101)]

    DIR_PATH = "output_generations_increased/"

    if os.path.isfile(DIR_PATH + "results.csv"):
        raise OSError("File exists")
    else:
        with open(DIR_PATH + "results.csv", "w") as file:
            file.write("max_generations, fitness_score, training_time\n")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(train_generations_increased, generations)
