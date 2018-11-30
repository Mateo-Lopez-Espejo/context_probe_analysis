import collections as coll
import os
import subprocess
import joblib as jl
import inspect
import hashlib
import nems.recording as nrec
import nems.signal as nsig


# ToDo Restructure cache in two functios, create and load.
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



def make_cache(function, func_args, classobj_name, recache=False, cache_folder='/home/mateo/mycache', use_hash=True, dump_func=jl.dump):
    '''
    caches the output of a function in a location defined by the function name and its arguments
    :param function: pointer to the function
    :param func_args: dict arguments to pass to the function
    :param recache: Bool. default(False). If True force run the function and recaches
    :param cache_folder: str. path to the location to save the cache files
    :user_hash: Bool. default(True). False uses a string composed by the function, and the used arguments. True, uses the
    python hash of the previous string,
    :return: path to the function output file.
    '''

    # creates cache folder if necesary
    if os.path.isdir(cache_folder):
        pass
    else:
        os.makedirs(cache_folder)

    # creates a unique name by hashing together the function, its argument (Todo and the git version)
    # gets all function parameters, updates with specified parameters, orders and makes into a string.
    signature = inspect.signature(function)
    funct_defaults = {k: v.default for k, v in signature.parameters.items()
                      if v.default is not inspect.Parameter.empty}

    funct_defaults.update(func_args)
    # excludes class objects like nems Recordings or Signals
    nems_objects = (nrec.Recording, nsig.SignalBase)

    funct_defaults = {key: val for key, val in funct_defaults.items() if not isinstance(val, nems_objects)}

    ordered_args = coll.OrderedDict(sorted(funct_defaults.items(), key=lambda t: t[0]))
    arg_string = ', '.join(['{}={}'.format(key, str(val)) for key, val in ordered_args.items()])

    # get the last commit that modified the file containing the function
    label = subprocess.check_output(['git', 'rev-parse', 'HEAD']) # todo is adding this really smart??

    unique_name = '{}-{}({})'.format(classobj_name, function.__name__, arg_string)
    name_hash = hashlib.sha1(unique_name.encode()).hexdigest()

    print('\nfunction call (aka unique name):{}\nhash: {}\n '.format(unique_name, name_hash))

    if use_hash is True:
        filename = os.path.join(cache_folder, name_hash)
    elif use_hash is False:
        filename = os.path.join(cache_folder, unique_name)
    else:
        raise ValueError('use_hash must be a boolean')

    # checks if a cache exists, runs function if necessary/specified
    if os.path.exists(filename) is False:
        print('cache not found at {}, running function'.format(filename))
        func_out = function(**func_args)
        print('cacheing to {}'.format(filename))
        dump_func(func_out, filename)
    else:
        print('cache found')
        if recache is True:
            print('force recache, running function')
            func_out = function(**func_args)
            print('cacheing to {}'.format(filename))
            dump_func(func_out, filename)
        elif recache is False:
            pass

    return filename


def get_cache(filename, cache_func=jl.load):
    print('loading cached file')
    return cache_func(filename)
