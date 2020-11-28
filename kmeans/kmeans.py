import math

class County:
    def __init__(self, name, values):
        self.name = name
        self.values = values
    def distance(self, othervals):
        dist = 0
        for i in range(len(self.values)):
            dist += abs(self.values[i]-othervals[i])
        return dist

class Cluster:
    def __init__(self):
        self.centroid = []
        self.contents = []
    def updateCentroid(self):
        self.centroid = [sum(k)/len(k) for k in zip(*[c.values for c in self.contents])]

    def names(self):
        names = ""
        for c in self.contents:
            names += c.name + "; "
        return names
    def clear(self):
        self.contents = []

def readData(filename):
    counties = []
    with open(filename) as f:
        lines = f.readlines()
        lines = [l[:-1] for l in lines]
        for i in range(1,len(lines)):
            line_l = lines[i].split(";")
            counties.append(County(line_l[0], [float(v) for v in line_l[1:]]))

    return counties

def initClusters(counties, num):
    clusters = []
    for i in range(num):
        newcluster = Cluster()
        newcluster.centroid = counties[i].values[:]
        clusters.append(newcluster)
    return clusters

def placeCounties(counties, clusters):
    for c in counties:
        score = math.inf
        final = None
        for clu in clusters:
            dist = c.distance(clu.centroid)
            if dist < score:
                score = dist
                final = clu
        final.contents.append(c)

def updateCentroids(clusters):
    for c in clusters:
        c.updateCentroid()

def clearClusters(clusters):
    for c in clusters:
        c.clear()

def writeOutput(clusters, filename):
    f = open(filename, "w")
    for i in range(len(clusters)):
        f.write("Cluster " + str(i+1) + "\n")
        f.write("size: " + str(len(clusters[i].contents)) + "\n")
        s = ", ".join([str(val) for val in clusters[i].centroid])
        f.write("[" + s + "]\n")
        f.write("; ".join([c.name for c in clusters[i].contents]) + ";\n\n")
    f.close()

def kmeans(infile, outfile, k, cycles):
    counties = readData(infile)
    clusters = initClusters(counties, k)
    for i in range(cycles):
        clearClusters(clusters)
        placeCounties(counties, clusters)
        updateCentroids(clusters)
    writeOutput(clusters, outfile)

kmeans("normalized_counties.txt", "output.txt", 30, 120)


