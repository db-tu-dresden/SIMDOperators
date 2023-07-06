import sys

from benchsuite import CMakeProject as cmp
import os
from matplotlib import pyplot as plt
import datetime

class MRun:
    operatorString = ""
    processingStyle = ""
    dataSize = 0
    offset = 0
    aligned = False
    durations = []
    
    def prepare(self):
        # sort durations
        self.durations.sort()
        # remove outliers
        self.durations = self.durations[1:-1]
        
    def average(self):
        return sum(self.durations) / len(self.durations)
    
    def mean(self):
        return self.durations[len(self.durations) // 2]

def readableSize(size) -> str:
    if size < 1024:
        return str(int(size)) + "B"
    elif size < 1024 * 1024:
        return str(int(size / 1024)) + "KB"
    elif size < 1024 * 1024 * 1024:
        return str(int(size / 1024 / 1024)) + "MB"
    else:
        return str(int(size / 1024 / 1024 / 1024)) + "GB"


class MBenchmark:
    # {operatorString -> {processingStyle -> {offset -> {aligned -> {dataSize -> durations}}}}
    cases = {}


def processBenchmark(csv_file, ignoreValues = {}):
    # parse csv file
    # structure:
    # operatorString, processingStyle, dataSize, offset, aligned, durations
    # durations: [duration1, duration2, ...]
    
    benchmark = MBenchmark()
    
    # read file
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            mrun = MRun()
            values = line.split(',')
            mrun.operatorString = values[0]
            mrun.processingStyle = values[1]
            mrun.dataSize = int(values[2])
            mrun.offset = int(values[3])
            mrun.aligned = True if values[4] == "1" else False
            mrun.durations = [float(x) for x in values[5:]]
            
            if mrun.processingStyle in ignoreValues["processingStyle"]:
                continue
            if mrun.offset in ignoreValues["offset"]:
                continue
            if mrun.aligned in ignoreValues["aligned"]:
                continue
            if mrun.dataSize in ignoreValues["dataSize"]:
                continue
            
            if mrun.operatorString not in benchmark.cases:
                benchmark.cases[mrun.operatorString] = {}
            if mrun.processingStyle not in benchmark.cases[mrun.operatorString]:
                benchmark.cases[mrun.operatorString][mrun.processingStyle] = {}
            if mrun.offset not in benchmark.cases[mrun.operatorString][mrun.processingStyle]:
                benchmark.cases[mrun.operatorString][mrun.processingStyle][mrun.offset] = {}
            if mrun.aligned not in benchmark.cases[mrun.operatorString][mrun.processingStyle][mrun.offset]:
                benchmark.cases[mrun.operatorString][mrun.processingStyle][mrun.offset][mrun.aligned] = {}
                
                
            mrun.prepare()
            # benchmark.cases[mrun.operatorString][mrun.processingStyle][mrun.offset][mrun.aligned][mrun.dataSize] = mrun.average()
            benchmark.cases[mrun.operatorString][mrun.processingStyle][mrun.offset][mrun.aligned][mrun.dataSize] = mrun.mean()
    
    return benchmark
    

def plotBenchmark_Curves(plotPath, benchmark):
    # plot benchmark as curves
    # x-axis: dataSize
    # y-axis: duration
    # different plots: operatorString
    # different curves: processingStyle, offset, aligned
    
    for operatorString in benchmark.cases:
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.xlabel("Data Size")
        plt.ylabel("Duration")
        # set title
        ax.set_title(operatorString)
        ax.grid(True)
        
        
        # get all data sizes
        dataSizes = []
        for processingStyle in benchmark.cases[operatorString]:
            for offset in benchmark.cases[operatorString][processingStyle]:
                for aligned in benchmark.cases[operatorString][processingStyle][offset]:
                    for dataSize in benchmark.cases[operatorString][processingStyle][offset][aligned]:
                        if dataSize not in dataSizes:
                            dataSizes.append(dataSize)
        ticks = []
        for i in range(len(dataSizes)):
            ticks.append(i)
            
        tickMapping = {}
        for (i, dataSize) in enumerate(dataSizes):
            tickMapping[dataSize] = i
        
        plt.xticks(ticks, [readableSize(ds) for ds in dataSizes])
        # logaritmic x-axis
        # plt.xscale("log")
        
        for processingStyle in benchmark.cases[operatorString]:
            for offset in benchmark.cases[operatorString][processingStyle]:
                for aligned in benchmark.cases[operatorString][processingStyle][offset]:
                    dataSizes = []
                    durations = []
                    curveName = processingStyle + "_" + str(offset) + "_" + str(aligned)
                    for dataSize in benchmark.cases[operatorString][processingStyle][offset][aligned]:
                        dataSizes.append(tickMapping[dataSize])
                        durations.append(benchmark.cases[operatorString][processingStyle][offset][aligned][dataSize])
                    plt.plot(dataSizes, durations, label=curveName)
                    
        plt.legend()
        plt.savefig(plotPath + "/" + operatorString + "_curve.png")
    

def plotBenchmark_Difference(plotPath, benchmark):
    # plot benchmark as curves
    # x-axis: dataSize
    # y-axis: duration
    # different plots: operatorString
    # different curves: processingStyle, offset, aligned
    
    
    for operatorString in benchmark.cases:
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.xlabel("Data Size")
        plt.ylabel("Difference (unaligned - aligned)")
        # set title
        ax.set_title(operatorString)
        ax.grid(True)
        
        
        # get all data sizes
        dataSizes = []
        for processingStyle in benchmark.cases[operatorString]:
            for offset in benchmark.cases[operatorString][processingStyle]:
                for aligned in benchmark.cases[operatorString][processingStyle][offset]:
                    for dataSize in benchmark.cases[operatorString][processingStyle][offset][aligned]:
                        if dataSize not in dataSizes:
                            dataSizes.append(dataSize)
        ticks = []
        for i in range(len(dataSizes)):
            ticks.append(i)
            
        tickMapping = {}
        for (i, dataSize) in enumerate(dataSizes):
            tickMapping[dataSize] = i
        
        plt.xticks(ticks, [readableSize(ds) for ds in dataSizes])
        # logaritmic x-axis
        # plt.xscale("log")
        
        for processingStyle in benchmark.cases[operatorString]:
            for offset in benchmark.cases[operatorString][processingStyle]:
                dataSizes = []
                durations = []
                curveName = processingStyle + "_" + str(offset)
                for (aligned, unaligned) in zip(benchmark.cases[operatorString][processingStyle][offset][True], benchmark.cases[operatorString][processingStyle][offset][False]):
                    dataSizes.append(tickMapping[aligned])
                    durations.append(benchmark.cases[operatorString][processingStyle][offset][False][unaligned] - benchmark.cases[operatorString][processingStyle][offset][True][aligned])
                    # for dataSize in benchmark.cases[operatorString][processingStyle][offset][aligned]:
                        # dataSizes.append(tickMapping[dataSize])
                        # durations.append(benchmark.cases[operatorString][processingStyle][offset][aligned][dataSize])
                plt.plot(dataSizes, durations, label=curveName)
                    
        plt.legend()
        plt.savefig(plotPath + "/" + operatorString + "_diff.png")


def main():

    # project_path = os.path.dirname(os.path.realpath(__file__))
    
    # preprocessing = cmp.CMakeProject("preprocessing", project_path + "/../../..", project_path + "/build")
    
    # preprocessing.switch_print_build_messages()
    # preprocessing.compile()

    # preprocessing.run()
    
    
    # timestamp for output file
    now = datetime.datetime.now()
    timestamp = now.strftime("%d_%m_%Y__%H_%M_%S")
    plotPath = "plots/" + timestamp
    os.makedirs(plotPath, exist_ok=True)
    
    
    
    ignoreValues = {}
    # ignoreValues["processingStyle"] = ["scalar"]
    ignoreValues["processingStyle"] = []
    # ignoreValues["offset"] = [1,3,5,6,7]
    ignoreValues["offset"] = []
    ignoreValues["aligned"] = []
    ignoreValues["dataSize"] = []
    # b = processBenchmark("build/test/benchmark/preprocessing/benchmark_19_06_2023__15_03_06.csv")
    # b = processBenchmark("build/test/benchmark/preprocessing/benchmark_19_06_2023__16_49_08.csv")
    # b = processBenchmark("build/test/benchmark/preprocessing/benchmark_19_06_2023__16_54_41.csv", ignoreValues)
    b = processBenchmark("build/test/benchmark/preprocessing/benchmark_19_06_2023__17_44_53.csv", ignoreValues)
    plotBenchmark_Curves(plotPath, b)
    plotBenchmark_Difference(plotPath, b)


if __name__ == "__main__":
    main()
