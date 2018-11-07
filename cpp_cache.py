import collections as coll
import os
import subprocess
import joblib as jl
import inspect


# ToDo Restructure cache in two functios, create and load.
'''
create should
'''





def set_name(name_args, signal_name=None, onlysig=False):
    '''
    creates a uniqe name using the key vals in name_args and an optional signal name. name_args sould be a dictionary
    of locals() excluding 'signal', and 'signame' itself
    :param name_args:
    :param signal_name:
    :return:
    '''

    ord_parms = coll.OrderedDict(sorted(name_args.items(), key=lambda t: t[0]))

    cache_name = '_'.join(['{}-{}'.format(key, str(val)) for key, val in ord_parms.items()])

    if signal_name != None:

        if onlysig == False:
            cache_name = '{}_{}'.format(signal_name, cache_name)

        elif onlysig == True:
            cache_name = signal_name

    return cache_name


def cache_wrap(obj_name, folder='/home/mateo/mycache', obj=None, recache=False):
    if os.path.isdir(folder):
        pass
    else:
        os.makedirs(folder)

    path = os.path.join(folder, obj_name)

    save_msg = 'Saving object at\n{}'.format(path)
    load_msg = 'Found cached object, loading from\n{}'.format(path)

    # forces save
    if recache == True:
        if obj != None:
            # resavese
            jl.dump(obj, path)
            print(save_msg)
            return obj
        else:
            print('Forced recaching, running...')
            return None

    elif recache == False:
        print('searching {}'.format(path))
        #
        if os.path.exists(path):
            if obj is not None:
                print('Object specified, skipping loading')
                return obj
            elif obj is None:
                print(load_msg)
                obj = jl.load(path)
                return obj

        elif not os.path.exists(path):
            if obj is None:
                print('Cache not found, running...')
                return None

            else:
                jl.dump(obj, path)
                print(save_msg)
                return obj



def make_cache(function, func_args, recache=False, cache_folder='/home/mateo/mycache', use_hash=True, dump_func=jl.dump):
    '''

    :param function:
    :param func_args:
    :param recache:
    :param cache_folder:
    :return:
    '''

    # creates cache folder if necesary
    if os.path.isdir(cache_folder):
        pass
    else:
        os.makedirs(cache_folder)

    # creates a unique name by hashing together the function, its argument and the git version
    # gets all function parameters, updates with specified parameters, orders and makes into a string.
    signature = inspect.signature(function)
    funct_defaults = {k: v.default for k, v in signature.parameters.items()
                      if v.default is not inspect.Parameter.empty}

    funct_defaults.update(func_args)
    ordered_args = coll.OrderedDict(sorted(funct_defaults.items(), key=lambda t: t[0]))
    arg_string = ', '.join(['{}={}'.format(key, str(val)) for key, val in ordered_args.items()])

    # get the last commit that modified the file containing the function
    label = subprocess.check_output(['git', 'rev-parse', 'HEAD']) # todo is adding this really smart??

    unique_name = '{}({})'.format(function.__name__, arg_string)
    name_hash = str(hash(unique_name))

    print('function call string:{}\nhash: {}'.format(unique_name, name_hash))

    if use_hash is True:
        filename = os.path.join(cache_folder, name_hash)
    elif use_hash is False:
        filename = os.path.join(cache_folder, unique_name)
    else:
        raise ValueError('use_hash must be a boolean')

    # checks if a cache exists, runs function if necessary/specified
    if os.path.exists(filename) is False:
        print('cache not found at {}, running function')
        func_out = function(**func_args)
        print('cacheing to {}'.format(filename))
        dump_func(func_out, filename)
    else:
        print('cache found')
        if recache is True:
            print('foce recache, running function')
            func_out = function(**func_args)
            print('cacheing to {}'.format(filename))
            dump_func(func_out, filename)
        elif recache is False:
            pass

    return filename


def get_cache(filename, cache_func=jl.load()):
    print('loading cached file')
    return jl.load(filename)
