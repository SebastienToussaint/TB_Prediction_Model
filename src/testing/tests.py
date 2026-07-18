from statistics import stdev

from scipy.stats import truncnorm



mean = 0

sd = 5

lower = -10
upper = 10


a = []

for i in range(10000):
    a.append(int(truncnorm.rvs(lower/sd, upper/sd, mean, sd, 1)[0]))

print(a)

for i in range(-10, 11):
    print((a.count(i)/len(a)*100))

