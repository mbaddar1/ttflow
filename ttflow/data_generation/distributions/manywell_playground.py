"""
This script is to get some hands on manywell distribution
"""
from ttflow.data_generation.distributions.many_well import ManyWell_Gaussian

if __name__ == "__main__":
    total_data_dim = 8
    manywell_dim = 4
    n = 10_000
    assert manywell_dim <= total_data_dim
    mw = ManyWell_Gaussian(dim=total_data_dim, many_well_dim=manywell_dim, normalised=True, use_gpu=False)
    x = mw.sample([n])
    ManyWell_Gaussian.pairwise_plot(sample=x, output_file_name="manywell.png")
