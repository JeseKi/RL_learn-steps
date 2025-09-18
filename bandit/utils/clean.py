import gc

def clear_var(*args):
    for var in args:
        del var
    gc.collect()