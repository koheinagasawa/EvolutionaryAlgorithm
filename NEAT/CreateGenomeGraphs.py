from GenomeVisualizer import OutputGenomes
import os

def CreateGraphsForGeneration(path, runIndex, generationIndex):
    fileName = path + "/Run" + str(runIndex) + "/Gen" + str(generationIndex) + ".json"
    OutputGenomes(fileName, -1, True, False)

def CreateGraphsForBestGenomes(path, runIndex):
    dirName = path + "/Run" + str(runIndex) + "/"

    for i in os.listdir(dirName):
        if ".json" in i:
            OutputGenomes(dirName + i, 1, True, False, dirName + "BestGenomes/")

dir = "XOR_Test/Results"

# Create graphs of all genomes in a generation of a run
CreateGraphsForGeneration(dir, 9, 58)

# Create graphs of all the best genomes throughout generations of a run
# numRun = 20
# for i in range(0, numRun - 1):
#     CreateGraphsForBestGenomes(dir, i)