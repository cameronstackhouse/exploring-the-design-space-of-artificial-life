""" 
Module to generate the dataset containing
information about genotypes and phenotypes produced
by evolutionary algorithms alongside their
behavioural properties and fitnesses.

The format for a xenobot listing is as follows in the JSON file:
[
    {
        "generation": int,
        "fitness_function": str,
        "evolutionary_algorithm": str,
        "genotype": 
        {
            "layers": 
            {
                "nodes": 
                {
                    "id": ,
                    "activation_function": ,
                    "connections_in": 
                    {
                        "id": int,
                    }   
                    "connections_out":
                    {
                        "id": int,
                    }
                }
                {
                
                }
            }
        },
        "phenotype":
        {
            "body": str,
            "num_muscle": int,
            "num_passive": int,
            "fitness": 
            {
                {"abs_movement": float}, 
                {traverse_obsticle": float},  
            }  
        }
    }
]  
"""

import json 

def write_json(filename, dict) -> None:
    """ 
    
    """
    json_object = json.dumps(dict)
    with open(filename, "w") as outfile:
        outfile.write(json_object)

def read_json(filename) -> dict:
    """ 
    
    """
    with open(filename, 'r') as openfile:
        json_object = json.load(openfile)
    
    return json_object

def add_xenobot(filename, xenobot: dict) -> None:
    """ 
    
    """
    with open(filename) as file:
        json_object = json.load(file)
    
    json_object.append(xenobot)
    
    with open(filename, 'w') as json_file:
        json.dump(json_object, json_file, indent=4, separators=(',', ': '))