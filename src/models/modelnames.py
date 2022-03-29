"""
This is just a repository for the main models that I have used, and some appropiate nicknames.
It helps me keep consisten model names across all my code and avoid silly bug hunts.
"""

# sanity check psth without jacknifes, it confirmed same response to probes past strf window
STRF_no_jk = "ozgf.fs100.ch18-ld-norm-epcpn.seq-avgreps_" \
             "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_" \
             "aev-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# STRF
STRF = "ozgf.fs100.ch18-ld-norm-epcpn.seq-avgreps_" \
       "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_" \
       "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# STRF long enoug to also capture the same time window used for population modulation
STRF_long = "ozgf.fs100.ch18-ld-norm-epcpn.seq-avgreps_" \
            "dlog-wc.18x1.g-fir.1x30-lvl.1-dexp.1_" \
            "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# pop mod PSTH
pop_mod = "ozgf.fs100.ch18-ld.popstate-dline.15.15.1-norm-epcpn.seq-avgreps_" \
          "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S.d_" \
          "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

## second generation models. with simplified nonlinearities for easier readout of the model parameterse
#
STRF_relu = "ozgf.fs100.ch18-ld-norm.l1-epcpn.seq-avgreps_" \
                 "wc.18x1.g-fir.1x15-lvl.1-relu.1_" \
                 "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

STRF_long_relu = "ozgf.fs100.ch18-ld-norm.l1-epcpn.seq-avgreps_" \
                 "wc.18x1.g-fir.1x30-lvl.1-relu.1_" \
                 "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# mean, corrected delayed lines
pop_mod_relu = "ozgf.fs100.ch18-ld.popstate-norm.l1-dline.15.15.1-epcpn.seq-avgreps_" \
               "wc.18x1.g-fir.1x15-lvl.1-relu.1-stategain.S.d_" \
               "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# for easy export
modelnames = {'STRF_no_jk': STRF_no_jk,
              'STRF': STRF,
              'STRF_long': STRF_long,
              'pop_mod': pop_mod,
              'STRF_long_relu': STRF_long_relu,
              'pop_mod_relu': pop_mod_relu}
