import xml.etree.ElementTree as ET

def read_sim_output(filename):
    """
    
    """
    #TODO
    tree = ET.parse(f"{filename}.xml")
    root = tree.getroot()
    for child in root:
        #TODO Can access different tags by child[index]
        pass