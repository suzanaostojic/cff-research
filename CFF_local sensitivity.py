#%%
import numpy as np
import pandas as pd
#%%
# Define CFF function and its partial derivatives


def cff(a, r1, r2, qp, qs_in, qs_out,  ev=156, ev_star=156, e_rec=10.2, e_rec_eol=7.63):
    # using the concrete industrial floor with GWP100
    # now qs_in = qsin/qp, qs_out = qs_out/qp
    return (1-r1)*ev + r1*(a*e_rec+(1-a)*ev*qs_in/qp) + (1-a)*r2*(e_rec_eol-ev_star*qs_out/qp)


def d_da(a, r1, r2, qp, qs_in, qs_out,  ev=156, ev_star=156, e_rec=10.2, e_rec_eol=7.63):
    return r1*(e_rec - ev*qs_in/qp)-r2*(e_rec_eol-ev_star*qs_out/qp)


def d_dr1(a, r1, r2, qp, qs_in, qs_out,  ev=156, ev_star=156, e_rec=10.2, e_rec_eol=7.63):
    return -ev + (a*e_rec + (1-a)*ev*qs_in/qp)


def d_dr2(a, r1, r2, qp, qs_in, qs_out,  ev=156, ev_star=156, e_rec=10.2, e_rec_eol=7.63):
    return (1-a)*(e_rec_eol-ev_star*qs_out/qp)


def d_dqs_in(a, r1, r2, qp, qs_in, qs_out,  ev=156, ev_star=156, e_rec=10.2, e_rec_eol=7.63):
    return r1*(1-a)*ev/qp


def d_dqs_out(a, r1, r2, qp, qs_in, qs_out,  ev=156, ev_star=156, e_rec=10.2, e_rec_eol=7.63):
    return -r2*(1-a)*ev_star/qp


#%%
# Define baseline values for each variable
a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0 = 0.5, 0, 0.5, 1, 0.5, 0.8  # concrete industrial floor

# Compute cff value at baseline
cff_0 = cff(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0)

# Compute absolute sensitivities at baseline
S1 = d_da(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0)
S2 = d_dr1(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0)
S3 = d_dr2(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0)
S4 = d_dqs_in(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0)
S5 = d_dqs_out(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0)

# Compute relative sensitivities at baseline
rS1 = d_da(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0) * a_0 / cff_0
rS2 = d_dr1(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0) * r1_0 / cff_0
rS3 = d_dr2(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0) * r2_0 / cff_0
rS4 = d_dqs_in(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0) * qs_in_0 / cff_0
rS5 = d_dqs_out(a_0, r1_0, r2_0, qp_0, qs_in_0, qs_out_0) * qs_out_0 / cff_0

# Store and rank the sensitivities
sensitivities = {"a": [S1, rS1], "r1": [S2, rS2], "r2": [S3, rS3], "qs_in": [S4, rS4], "qs_out": [S5, rS5]}

sensitivities_df = pd.DataFrame.from_dict(sensitivities, orient='index')
print(sensitivities_df)

#######################################################
# To be checked r1 absolute sensitivity is different between code and excel
# Maybe would be interesting now to run a MC analysis of the partial derivatives to check how much
# absolute sensitivities evolve in the variable space
#######################################################