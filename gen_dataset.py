""" 
Module to generate the dataset containing
information about genotypes and phenotypes produced
by evolutionary algorithms alongside their
behavioural properties and fitnesses
"""

#TODO Maybe use JSON file, hard to represent NN otherwise

import csv
from typing import List

def initilise_file(filename: str) -> None:
    with open(filename, mode="w") as xenobot_data_file:
        file_writer = csv.writer(xenobot_data_file, delimiter=',')
        file_writer.writerow(["Genotype", "Xenobot", "Fitness",  "Experiment", "Frequency1", "Frequency2", "Frequency3", "Motif1"])

def write_to(filename: str, data: List) -> None:
    with open(filename, mode="a") as xenobot_file:
        file_writer = csv.writer(xenobot_file, delimiter=',')
        file_writer.writerow(data)

def initilise_hyperneat_file(filename: str) -> None:
    with open(filename, mode="w") as xenobot_data_file:
        file_writer = csv.writer(xenobot_data_file, delimiter=',')
        file_writer.writerow(["CPPN", "Genotype", "Xenobot", "Fitness",  "Experiment", "Frequency1", "Frequency2", "Frequency3", "Motif1"])

if __name__ == "__main__":
    filename = "test.csv"
    #initilise_file(filename)
    write_to(filename, ["test", "test2", "0.0", "Test3", "Locomotion"])