import concurrent.futures
from definition import *

if __name__ == "__main__":
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [60, 300, 600]
        results = executor.map(train_once, secs)

        for result, seconds in zip(results, secs):
            print("-="*20, seconds, "-="*20)
            for r in result:
                print(r)
            print('\n\n')
    
# print(results)