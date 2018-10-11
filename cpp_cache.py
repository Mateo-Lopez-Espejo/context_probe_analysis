import collections as coll
import os

import joblib as jl


# todod figure out how to chach

def set_name(name_args, signal_name=None, onlysig=False):
    '''
    creates a uniqe name using the key vals in name_args and an optional signal name. name_args sould be a dictionary
    of locals() excluding 'signal', and 'signame' itself
    :param name_args:
    :param signal_name:
    :return:
    '''

    ord_parms = coll.OrderedDict(sorted(name_args.items(), key=lambda t: t[0]))

    cache_name = '_'.join(['{}:{}'.format(key, str(val)) for key, val in ord_parms.items()])

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
