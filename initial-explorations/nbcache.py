import pickle
import os
import functools


def cacheable(cache_file, use_cache=True, verbose=True):
    '''
    Cache a function's return into a pickle. On subsequent calls, re-use the cache.
    '''
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # if the cache exists and we want to use it,
            # simply load and return
            if os.path.exists(cache_file) and use_cache:
                if verbose:
                    print(f'Loading from cache {cache_file}...')
                return pickle.load(open(cache_file, 'rb'))
            
            # cache does not exist or is disabled, evaluate
            # target function
            if verbose:
                print(f'Evaluating function...')
            out = func(*args, **kwargs)

            # if caching enabled, save it to disk
            if use_cache:
                if verbose:
                    print(f'Caching to {cache_file}...')
                pickle.dump(out, open(cache_file, 'wb'))
            return out
        return wrapper
    return inner


if __name__ == '__main__':
    @cacheable('test.pickle', use_cache=False)
    def test_func():
        # simulate work
        import time
        time.sleep(5)
        return 'ok'

    print('Running test once...')
    print('\t' + test_func())
    print('Running test twice...')
    print('\t' + test_func())
