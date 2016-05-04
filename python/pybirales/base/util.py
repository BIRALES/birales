from multiprocessing import Lock

print_lock = Lock()

def atomic_print(str):
    with print_lock:
        print str
