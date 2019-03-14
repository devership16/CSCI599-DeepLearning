import jellyfish
import time
from difflib import SequenceMatcher

def getStringSimilarity(s1, s2):
    return jellyfish.jaro_winkler(s1, s2)

def getStringsimilarity_difflib(s1,s2):
    return SequenceMatcher(None, s1, s2).ratio()

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
        fout.write( key + " -> ( "+ data[key][0] +", " + str(data[key][1])+" )\n")

    fout.close()
    return

def getMostSimilar(data):
    mostSimilar = dict()  # key-> Target: Value-> Most Similar Name, Sim Value
    
    for i, name1 in enumerate(data): 
        maxSim = -float('inf')
        mostSimilar[name1] = ["", maxSim]
        
        for j, name2 in enumerate(data):
            
            if i==j or name1==name2:
                continue
            
            sim1 = getStringSimilarity(name1, name2) 
#             sim2 = getStringsimilarity_difflib(name1, name2)
            
            sim = sim1
            
            if sim > mostSimilar[name1][1]:
                mostSimilar[name1][1] = sim
                mostSimilar[name1]=[name2, sim]
    
    return mostSimilar



if __name__=='__main__':   
    
    start=time.time()
    inPath = "../../resources/datasets/"
    outPath = "../../resources/results/"
    
    data = getInputData(inPath+"names.txt")
    mostSimilar = getMostSimilar(data)
    print(mostSimilar[:10])
    print ("\nRun time: "+ str(time.time()-start)+" seconds" )  
    printResults(outPath + "mostSimilarName.txt", mostSimilar)
    