from scipy.stats.distributions import chi2

a = (1-0.997)
x_len = 3
N = 10*4

r1 = (1/N)*chi2.ppf(a/2.0, df=N*x_len)
r2 = (1/N)*chi2.ppf(1-a/2.0, df=N*x_len)

print(r1)
print(r2)