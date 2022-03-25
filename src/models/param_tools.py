import numpy as np

from nems.xform_helper import load_model_xform


def get_population_weights(cellid, batch, modelname):

    xfspec, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=False)

    ms = ctx['modelspec']

    # find the postion of the stategain module
    modules = str(ms).split('\n')
    idx = modules.index('nems.modules.state.state_dc_gain')

    _ = ms.set_cell(0)
    _ = ms.set_fit(0)

    chn, npop = ms.phi[idx]['d'].shape

    mean_pop_gain = np.empty((ms.jack_count, npop))

    for jc in range(ms.jack_count):
        _ = ms.set_jack(jc)
        mean_pop_gain[jc, :] = ms.phi[idx]['d'][0,:] # drops cell first singleton dimesion

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(mean_pop_gain, aspect='auto')
    fig.show()

    mean_pop_gain = mean_pop_gain.mean(axis=0)

    return mean_pop_gain


if __name__ == '__main__':
    cellid = 'TNC014a-22-2'
    batch = 326

    modelname = "ozgf.fs100.ch18-ld.popstate-dline.15.15.1-norm-epcpn.seq-avgreps_" \
                "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S.d_" \
                "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

    mean_pop_gain = get_population_weights(cellid=cellid, batch=batch, modelname=modelname)

    import matplotlib.pyplot as plt
    plt.plot(mean_pop_gain)
    plt.show()


