import jellyfish

def getStringSimilarity(s1, s2):
    return jellyfish.jaro_winkler(s1, s2)


# s1 = "Jaro-Winkler is a modification/improvement to Jaro distance, like Jaro it gives a floating point response in [0,1] where 0 represents two completely dissimilar strings and 1 represents identical strings."
# s2 = "Jaro-winler"
# print(getStringSimilarity(s1, s2))
    


