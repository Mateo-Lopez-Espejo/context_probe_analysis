def hola(foo=1, bar=2, baz=3):


    out = '{} {} {}'.format(foo, bar, baz)

    return out


def mundo(eggs='a', hamm='b'):

    print(hola())

    out = '{} {}'.format(eggs, hamm)

    return out


def test(foo=1, bar='a'):

    baz = True

    print(locals())
    # {'foo':1, 'bar':'a', 'baz':True, ...}

    print(func_args())
    # {'foo':1, 'bar':'a'}
