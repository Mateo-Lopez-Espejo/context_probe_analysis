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

# mean introspection to prior activity
self_mod_relu = "ozgf.fs100.ch18-ld-norm.l1-dline.15.15.1.i.resp.o.state-epcpn.seq-avgreps_" \
                "wc.18x1.g-fir.1x15-lvl.1-relu.1-stategain.S.d_" \
                "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# mean past population activity modulating strf response
pop_mod_relu = "ozgf.fs100.ch18-ld.popstate-norm.l1-dline.15.15.1-epcpn.seq-avgreps_" \
               "wc.18x1.g-fir.1x15-lvl.1-relu.1-stategain.S.d_" \
               "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# Only use the self response as a predictor
self_lone_relu = "ozgf.fs100.ch18-ld-norm.l1-dline.15.15.1.i.resp.o.stim-epcpn.seq-avgreps_" \
                 "wc.Nx1-fir.1x1-lvl.1-relu.1_" \
                 "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# Only use the population response as a predictor
pop_lone_relu = "ozgf.fs100.ch18-ld.popstate-norm.l1-dline.15.15.1.i.state.o.stim-epcpn.seq-avgreps_" \
                "wc.Nx1-fir.1x1-lvl.1-relu.1_" \
                "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# STP rank 1
STP_STRF1_relu = "ozgf.fs100.ch18-ld-norm.l1-epcpn.seq-avgreps_" \
                 "wc.18x1.g-stp.1.q-fir.1x15-lvl.1-relu.1_" \
                 "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# STP rank 2
STP_STRF2_relu = "ozgf.fs100.ch18-ld-norm.l1-epcpn.seq-avgreps_" \
                 "wc.18x2.g-stp.2.q-fir.2x15-lvl.1-relu.1_" \
                 "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

## Third generation models. have matched sets of parameters, and shuffle parts (self, pop, all) as necesary
# these turned out to have a bad bias, the sbb stands for shuffled bad bias, and lack state gain normalization

match_sbb_STRF = "ozgf.fs100.ch18-ld.popstate-norm.l1-epcpn.seq-avgreps-shfcat.i.resp0.state0.o.state-dline.15.15.1_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_sbb_pop = "ozgf.fs100.ch18-ld.popstate-norm.l1-epcpn.seq-avgreps-shfcat.i.resp0.state.o.state-dline.15.15.1_" \
            "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
            "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_sbb_self = "ozgf.fs100.ch18-ld.popstate-norm.l1-epcpn.seq-avgreps-shfcat.i.resp.state0.o.state-dline.15.15.1_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_bb_full = "ozgf.fs100.ch18-ld.popstate-norm.l1-epcpn.seq-avgreps-shfcat.i.resp.state.o.state-dline.15.15.1_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# quick test of rolled instead of shuffled regressors, r stands for rolled,
# these also had the wrong bias. and lack state gain normalization
match_rbb_STRF = "ozgf.fs100.ch18-ld.popstate-norm.l1-epcpn.seq-avgreps-shfcat.i.resp1.state1.o.state-dline.15.15.1_" \
              "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
              "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_rbb_pop = "ozgf.fs100.ch18-ld.popstate-norm.l1-epcpn.seq-avgreps-shfcat.i.resp1.state.o.state-dline.15.15.1_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_rbb_self = "ozgf.fs100.ch18-ld.popstate-norm.l1-epcpn.seq-avgreps-shfcat.i.resp.state1.o.state-dline.15.15.1_" \
              "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
              "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# old incorrect order models, epcpn and avgreps happens after rolling/shufflign. rbo stands for rolled bad order
match_rbo_STRF = "ozgf.fs100.ch18-ld.popstate-norm.l1-shfcat.i.resp1.state1.o.state-dline.15.15.1-epcpn.seq-avgreps_" \
              "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
              "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_rbo_pop = "ozgf.fs100.ch18-ld.popstate-norm.l1-shfcat.i.resp1.state.o.state-dline.15.15.1-epcpn.seq-avgreps_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_rbo_self = "ozgf.fs100.ch18-ld.popstate-norm.l1-shfcat.i.resp.state1.o.state-dline.15.15.1-epcpn.seq-avgreps_" \
              "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
              "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_rbo_full = "ozgf.fs100.ch18-ld.popstate-norm.l1-shfcat.i.resp.state.o.state-dline.15.15.1-epcpn.seq-avgreps_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"


# model with rolling, correct state gain normalization, and no erroneous extra bias/baseline
match_STRF = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp1.state1.o.state.mm-dline.15.15.1-norm.l1_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_pop = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp1.state.o.state.mm-dline.15.15.1-norm.l1_" \
            "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
            "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_self = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp.state1.o.state.mm-dline.15.15.1-norm.l1_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

match_full = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp.state.o.state.mm-dline.15.15.1-norm.l1_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# test
matcht_STRF = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp1.state1.o.state.mm-dline.15.15.1-norm.l1_" \
             "wc.18x1.g-fir.1x15-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10.o-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"


# model with rolling, correct state gain normalization, and no erroneous extra bias/baseline. Extende STRF
matchl_STRF = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp1.state1.o.state.mm-dline.15.15.1-norm.l1_" \
             "wc.18x1.g-fir.1x30-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

matchl_pop = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp1.state.o.state.mm-dline.15.15.1-norm.l1_" \
            "wc.18x1.g-fir.1x30-lvl.1-stategain.S.d-relu.1_" \
            "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

matchl_self = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp.state1.o.state.mm-dline.15.15.1-norm.l1_" \
             "wc.18x1.g-fir.1x30-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

matchl_full = "ozgf.fs100.ch18-ld.popstate-epcpn.seq-avgreps-shfcat.i.resp.state.o.state.mm-dline.15.15.1-norm.l1_" \
             "wc.18x1.g-fir.1x30-lvl.1-stategain.S.d-relu.1_" \
             "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"


# for easy export
modelnames = {'STRF_no_jk': STRF_no_jk,
              'STRF': STRF,
              'STRF_long': STRF_long,
              'pop_mod': pop_mod,
              'STRF_relu': STRF_relu,
              'STRF_long_relu': STRF_long_relu,
              'pop_mod_relu': pop_mod_relu,
              'self_mod_relu': self_mod_relu,
              'pop_lone_relu': pop_lone_relu,
              'self_lone_relu': self_lone_relu,
              'STP_STRF1_relu': STP_STRF1_relu,
              'STP_STRF2_relu': STP_STRF2_relu,

              'match_sbb_STRF': match_sbb_STRF,
              'match_sbb_self': match_sbb_self,
              'match_sbb_pop': match_sbb_pop,
              'match_bb_full': match_bb_full,

              'match_rbb_STRF': match_rbb_STRF,
              'match_rbb_self': match_rbb_self,
              'match_rbb_pop': match_rbb_pop,

              'match_rbo_STRF': match_rbo_STRF,
              'match_rbo_self': match_rbo_self,
              'match_rbo_pop': match_rbo_pop,
              'match_rbo_full': match_rbo_full,

              'match_STRF': match_STRF,
              'match_self': match_self,
              'match_pop': match_pop,
              'match_full': match_full,

              'matchl_STRF': matchl_STRF,
              'matchl_self': matchl_self,
              'matchl_pop': matchl_pop,
              'matchl_full': matchl_full,

              'matcht_STRF': matcht_STRF,
              }
