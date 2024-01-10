"""
Mehran Ali Banka - Sep 2023
----------------------------

This file contains the constants variables
that are used during the exectution of this code

"""

# Path to the test documents
# This is the root folder to store test cases, and generated results after testing.
# The test case should be individual txt files like 0.txt, 1.txt and so on..
# They should be under document_root + "/Informational"
# For example: C:/Search_Engines/Final_Project/Data/Informational/0.txt
# Please refer to the supplied test case to see how to make new test cases
document_root = r"C:/Search_Engines/Final_Project/Data"

# Path where the results of run_all_tests is stored
test_results_file = r"C:/Search_Engines/Final_Project/Data/Informational/test_results.txt"

# alpha (weight given to semantic similarity)
alpha = 0.90

# beta ( weight to control sentence-to-sentence sim vs sentence-to-query sim)
beta = 0.15

# max iterations to solve the markov chain
kmax = 10

# max no of sentences to extract from the extractive summarizer
max_sentences_to_extract = 10

# Show graph viz's when doing QSLK
show_viz = False