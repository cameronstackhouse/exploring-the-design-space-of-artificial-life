import xml.etree.ElementTree as ET

def read_sim_output(filename):
    """
    
    """
    #TODO
    results = []
    tree = ET.parse(f"{filename}.xml")
    root = tree.getroot()
    for child in root:
        individual_result = dict()
        #TODO Can access different tags by child[index]
        results.append(individual_result)

    return results