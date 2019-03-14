import jellyfish
import time

def getStringSimilarity(s1, s2):
    return jellyfish.jaro_winkler(s1, s2)

def getInputData(path):
    
    fin = open(path, "r", encoding="utf8")
    data = []
    for line in fin:
        line = line.strip()
        data.append(line)
    
    fin.close()
#     print(len(data))
    return data

def printResults(path, data):
    
    fout = open(path, "w+", encoding = "utf8" )
    
    for key in data.keys():
        fout.write( key + ", "+ data[key][0] +": " + str(data[key][1])+"\n")

    fout.close()
    return

def getMostSimilar(data):
    mostSimilar = dict()  # key-> Target: Value-> Most Similar Name, Sim Value
    
    for i, name1 in enumerate(data): 
        maxSim = -float('inf')
        for j, name2 in enumerate(data):
            if i==j:
                continue
            sim = getStringSimilarity(name1, name2) 
            if sim > maxSim:
                maxSim = sim
                mostSimilar[name1]=(name2, sim)
    
    return mostSimilar

if __name__=='__main__':   
    
    start=time.time()
    inPath = "../../resources/datasets/"
    outPath = "../../resources/results/"
    
    data = getInputData(inPath+"names.txt")
    mostSimilar = getMostSimilar(data)
    print ("\nRun time: "+ str(time.time()-start)+" seconds" )  
    printResults(outPath + "mostSimilarName.txt", mostSimilar)
    


