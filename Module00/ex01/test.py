from TinyStatistician import TinyStatistician
import numpy
tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]

print("mean a = ", tstat.mean(a))
print("empty mean = ", tstat.mean([]))
print("scalar data mean = ", tstat.mean(2))
print("list of strngs mean = ", tstat.mean(["o", "hello"]))

print("median a = ", tstat.median(a))
print("even list median = ", tstat.median([1,3,2,6]))
print("numpy even median = ", numpy.median([1,3,2,6]))
print("empty median = ", tstat.median([]))
print()


print("quartiles a = ", tstat.quartile(a))
print("empty quartile = ", tstat.quartile([]))
print("even quartiles = ",numpy.quantile([1, 42, 10, 59], [0.25, 0.75]))
print("numpy even quartiles = ",numpy.quantile([1, 42, 10, 59], [0.25, 0.75]))
print()

print("variance a = ", tstat.var(a))
print("empty variance = ", tstat.var([]))
print()

print("standard deviation a = ", tstat.std(a))
print("empty standard deviation = ", tstat.std([]))
print()