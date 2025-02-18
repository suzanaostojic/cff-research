#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

#%%
# Step 1: Define the distributions for parameters
# Assume parameters follow normal distributions with given means and standard deviations

# ev_mean, ev_std = 0.2, 0.4
# ev_star_mean, ev_star_std = 0.2, 0.4
# e_rec_mean, e_rec_std = 0.2, 0.4
# e_rec_eol_mean, e_rec_eol_std = 0.2, 0.4

# using case study 1, concrete industrial floor with GWP100
a_mean, a_std = 0.6, 0.3
r1_mean, r1_std = 0, 0.6
r2_mean, r2_std = 0.5, 0.5
qs_in_mean, qs_in_std = 0.2, 0.4
qs_out_mean, qs_out_std = 0.8, 0.1

n_sim = 10000000

#%%
# Step 2: Define the CFF function

##############################################################################
# I think I am missing the disposal part? I only have the material CFF part
##############################################################################

def cff(a, r1, r2, qs_in, qs_out, ev=156, ev_star=156, e_rec=10.2, e_rec_eol=7.63):
    # using the concrete industrial floor with GWP100
    # now qs_in = qsin/qp, qs_out = qs_out/qp
    return (1-r1)*ev + r1*(a*e_rec+(1-a)*ev*qs_in) + (1-a)*r2*(e_rec_eol-ev_star*qs_out)



#%%
# Step 3: Define random samples for each parameter


def truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std  # Convert bounds to standard normal space
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


# Function to generate constrained fr, fu samples
def sample_constrained_norm(mean1, std1, mean2, std2, n):
    fr = truncated_normal(mean1, std1, 0, 1, n)  # Ensure fr is between 0 and 1
    fu = np.array(
        [truncated_normal(mean2, std2, 0, 1 - fr[i], 1)[0] for i in range(n)])  # Adjust upper bound dynamically

    return fr, fu


# Generate samples for constrained variables

a_samples = np.random.uniform(0.4, 0.6, n_sim)
r1_samples = np.random.uniform(0, 0.2, n_sim)
r2_samples = np.random.uniform(0,1, n_sim)
qs_in_samples = np.random.uniform(0,1, n_sim)
qs_out_samples = np.random.uniform(0.7,0.9, n_sim)

'''
a_samples = truncated_normal(a_mean, a_std, 0, 1, n_sim)
r1_samples = truncated_normal(r1_mean, r1_std, 0, 1, n_sim)
r2_samples = truncated_normal(r2_mean, r2_std, 0, 1, n_sim)
qs_in_samples = truncated_normal(qs_in_mean, qs_in_std, 0, 1, n_sim)
qs_out_samples = truncated_normal(qs_out_mean, qs_out_std, 0, 1, n_sim)
'''

#%%
# Verify designed samples

print(f"Max: {np.max(a_samples)}")
print(f"Min: {np.min(a_samples)}")
print(f"Mean: {np.mean(a_samples)}")
print(f"Standard Deviation: {np.std(a_samples)}")
print(f"Median: {np.median(a_samples)}")
print(f"95th Percentile: {np.percentile(a_samples, 95)}")
#%%
# Step 4: Calculate CFF for each set of samples

cff_samples = cff(a_samples, r1_samples, r2_samples,
                  qs_in_samples, qs_out_samples)

#%%
# Step 5: Analyze and visualize the results
# Plot histogram of cff_samples

plt.figure(figsize=(10, 6))
plt.hist(cff_samples, bins=50, edgecolor='black', alpha=0.7)
plt.title('Monte Carlo Simulation Results for CFF')
plt.xlabel('Value of CFF')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig("Monte-Carlo results.png")
plt.show()

#%%
# Step 6: Calculate and display statistics

print(f"Max of CFF: {np.max(cff_samples)}")
print(f"Min of CFF: {np.min(cff_samples)}")
print(f"Mean of CFF: {np.mean(cff_samples)}")
print(f"Standard Deviation of CFF: {np.std(cff_samples)}")
print(f"Median of CFF: {np.median(cff_samples)}")
print(f"95th Percentile of CFF: {np.percentile(cff_samples, 95)}")