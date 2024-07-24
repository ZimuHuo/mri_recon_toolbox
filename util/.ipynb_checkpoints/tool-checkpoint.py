


t1t2dict = {
    "csf": [4000, 2000],
    "graymatter": [900, 90],
    "fat": [250, 70],
    "heart":[1300, 50],
   } 
def t1t2(tissue):
    return t1t2dict[tissue]
t2dict = {
    "graymatter": 90,
    "deep_graymatter": 80,
    "whitematter": 60,
    "csf": 2000,
   } 
def t2(tissue):
    return t2dict[tissue]