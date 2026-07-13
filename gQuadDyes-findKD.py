import statistics
from scipy import stats

def formatData(rawStr):
    return [float(value) for value in rawStr.split()]

def splitTrials(dataList, numTrials):
    fin = []
    trialSize = len(dataList)//numTrials
    for i in range(numTrials):
        start = i * trialSize
        end = start + trialSize
        fin.append(dataList[start:end])
    return fin

def arrangeData(fluro):
    standData = []
    for val in fluro:
        standData.append(val - fluro[0])
    return standData

def predictFluro(Bmax, X, Kd):
    return (Bmax * X) / (X + Kd)

def findKd(conc, fluro, learningRate=0.00001, iteration=100000):
    Bmax = max(fluro)
    Kd = 1.0
    n = len(conc)

    for step in range(iteration):
        dBmax = 0
        dKd = 0
        for i in range(n):
            X = conc[i]
            Y = fluro[i]
            pred = predictFluro(Bmax, X, Kd)
            dev = Y - pred

            dBmax += -2 * dev * (X / (X + Kd))
            dKd += 2 * dev * ((Bmax * X) / (X + Kd) ** 2)
    
        Bmax -= learningRate * (dBmax/n)
        Kd -= learningRate * (dKd/n)

    return Bmax, Kd

finConc = [0, 1, 5, 10, 15, 25]
#finConc = formatData('0	0.25	0.5	1	2	4	6	8	10	15	20	30')
allRawFluro = '0.03710	0.1685	0.3745	0.5086	0.4397	0.7279  0.004551	0.2219	0.3003	0.4088	0.6467	0.6933  0.01853	0.1894	0.4834	0.6820	0.8038	0.6935     0.02115	0.2141	0.4812	0.5257	0.7292	0.6763  0.03981	0.2919	1.179	1.216	1.501	1.546   0.02019	0.4489	0.7804	1.280	1.480	1.364'

allFluroList = formatData(allRawFluro)
finTrialsList = splitTrials(allFluroList, 6)
finData = [arrangeData(trial) for trial in finTrialsList]

numConditions = 3
kds = [[] for i in range(numConditions)]

for i in range(len(finData)):
    trial = finData[i]
    condition = i // 2
    trialNum = i % 2

    Bmax, Kd = findKd(finConc, trial)
    kds[condition].append(Kd)

    print(f'Condition {condition + 1}/Trial {trialNum + 1}: Bmax = {Bmax}, Kd = {Kd}')

for i in range(numConditions):
    kdList = kds[i]
    mean = statistics.mean(kdList)
    sd = statistics.stdev(kdList)
    print(f'Condition {i+1}: Mean Kd = {mean}, Standard Deviation = {sd}')

#tStat, pVal = stats.ttest_ind(kds[0], kds[1])
tStat, pVal = stats.f_oneway(kds[0], kds[1], kds[2])
print(f'p-value = {pVal}')
