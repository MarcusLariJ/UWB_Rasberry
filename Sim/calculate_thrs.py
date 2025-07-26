from scipy.stats.distributions import chi2

a = (1-0.997)
x_len = 3

r1 = chi2.ppf(a/2.0, df=x_len)
r2 = chi2.ppf(1-a/2.0, df=x_len)

print(r1)
print(r2)