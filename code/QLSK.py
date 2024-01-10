"""
Mehran Ali Banka - Sep 2023
----------------------------
This code implements the QLSK algorithm
for snippet generation

"""

import Constants as cons
import nlp_worker as nlpw
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import re
import Abstractive_summarizer as abs2

def extract_query(sample_path):
    # Read the content of the file
    with open(sample_path, 'r',encoding='utf-8', errors='ignore') as file:
        content = file.read()

    # Extract the sentence between <q> and </q> tags
    start_tag = '<q>'
    end_tag = '</q>'

    # Use regular expressions to find the content between the tags
    match = re.search(f'{re.escape(start_tag)}(.*?){re.escape(end_tag)}', content, re.DOTALL)

    if match:
        extracted_sentence = match.group(1).strip()
        return extracted_sentence.lower()
    else:
        print("No match found between <q> and </q> tags.")
        return None

def extract_doc(sample_path):
    # Read the content of the file
    with open(sample_path, 'r',encoding='utf-8', errors='ignore') as file:
        content = file.read()

    # Extract all sentences between <d> and </d> tags
    start_tag = '<d>'
    end_tag = '</d>'

    # Use regular expressions to find all content between the tags
    matches = re.findall(f'{re.escape(start_tag)}(.*?){re.escape(end_tag)}', content, re.DOTALL)

    for match in matches:
        extracted_sentence = match.strip()
        return extracted_sentence

def extract_summary(sample_path):
    # Read the content of the file
    with open(sample_path, 'r',encoding='utf-8', errors='ignore') as file:
        content = file.read()

    # Extract all sentences between <summ> and </summ> tags
    start_tag = '<summ>'
    end_tag = '</summ>'

    # Use regular expressions to find all content between the tags
    matches = re.findall(f'{re.escape(start_tag)}(.*?){re.escape(end_tag)}', content, re.DOTALL)

    for match in matches:
        extracted_sentence = match.strip()
        return extracted_sentence


def create_s2s_graph(processed_sentences_with_ids,word_set_dict):
    # Create an undirected graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(processed_sentences_with_ids.keys())

    # Add edges with weights
    for source_id, source_label in processed_sentences_with_ids.items():
        for target_id, target_label in processed_sentences_with_ids.items():
                weight = get_s2s_weight(source_id,target_id,word_set_dict)
                G.add_edge(source_id, target_id, weight=weight)
                G.add_edge(target_id, source_id, weight=weight)

    #Draw the graph
    if(cons.show_viz):            
        pos = nx.spring_layout(G) 
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black")

        # Add edge labels (weights)
        edge_labels = {(i, j): round(G[i][j]["weight"],2) for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

        # Show the plot
        plt.show()
        adj_matrix = nx.adjacency_matrix(G).todense()
        # To show as a matrix
        # Display the adjacency matrix using matplotlib
        plt.imshow(adj_matrix, cmap='viridis', interpolation='none')

        plt.title('Graph Adjacency Matrix')
        plt.colorbar(label='Edge Existence')
        plt.xticks(np.arange(len(G.nodes())), labels=list(G.nodes()))
        plt.yticks(np.arange(len(G.nodes())), labels=list(G.nodes()))
        plt.show()
        return G

def get_s2s_weight(src_id, target_id,word_set_dict):
    # Root words of sentence 1
    src_set = set(word_set_dict[src_id])
    # Root words of sentence 2
    dest_set = set(word_set_dict[target_id])
    # Root words in list - as we need index for syntactic similarity
    src_list = word_set_dict[src_id]
    dest_list = word_set_dict[target_id]
    # global set (union of the two)
    word_set = src_set.union(dest_set)
    # create the semantic vectors
    semantic_vector_1 = []
    semantic_vector_2 = []
    # create the syntactic vector
    syntactic_vector_1 = []
    syntactic_vector_2 = []

    for word in word_set:
        # compute the cell score for sentence 1
        if word in src_set:
            # set the cell score to 1 if the sentence has the word
            semantic_vector_1.append(1)
            # append the location of the word in sentence 1
            syntactic_vector_1.append(src_list.index(word))                        
        else:
            max_sim_score, max_idx = nlpw.content_word_expansion_score(word,src_list)
            semantic_vector_1.append(max_sim_score)
            syntactic_vector_1.append(max_idx)
        # compute the cell score for sentence 2
        if word in dest_set:
            # set the cell score to 1 if the sentence has the word
            semantic_vector_2.append(1)
            # append the location of the word in sentence 2
            syntactic_vector_2.append(dest_list.index(word))
        else:
            max_sim_score, max_idx = nlpw.content_word_expansion_score(word,dest_list)
            semantic_vector_2.append(max_sim_score)  
            syntactic_vector_2.append(max_idx)  
    
    semantic_score = calc_semantic_score(semantic_vector_1,semantic_vector_2)
    syntactic_score = calc_syntactic_score(syntactic_vector_1,syntactic_vector_2)
    return cons.alpha*semantic_score + (1 - cons.alpha)*syntactic_score

# Calculate the syntactic similarity between two sentences  
def calc_syntactic_score(syntactic_vector_1,syntactic_vector_2):
    # Compute the Euclidean distance between O1 and O2
    #distance = sum((a - b) ** 2 for a, b in zip(syntactic_vector_1, syntactic_vector_2)) ** 0.5
    # Compute the sum of magnitudes of O1 and O2
    #magnitude_sum = sum(a + b for a, b in zip(syntactic_vector_1, syntactic_vector_2))
    # Calculate the similarity using the provided formula
    #similarity = 1 - distance / magnitude_sum
    #return similarity
    try:
        O1 = np.array(syntactic_vector_1)
        O2 = np.array(syntactic_vector_2)
        # Calculate the Euclidean norms
        norm_diff = np.linalg.norm(O1 - O2)
        norm_sum = np.linalg.norm(O1 + O2)
        # Calculate the similarity score
        if(norm_sum == 0): return 0
        similarity_score = 1 - (norm_diff / norm_sum)
    except Exception as e:
        print("Vec 1: ", syntactic_vector_1)
        print("Vec 2: ", syntactic_vector_2)
        return 0
    return similarity_score

# Calculate the semantic similarity between two sentences
def calc_semantic_score(semantic_vector_1,semantic_vector_2):
    # calculate the numerator
    numerator = sum(x * y for x, y in zip(semantic_vector_1, semantic_vector_2))
    # calculate the denominator
    vector_1_norm = math.sqrt(sum(x ** 2 for x in semantic_vector_1))
    vector_2_norm = math.sqrt(sum(x ** 2 for x in semantic_vector_2))
    denominator = vector_1_norm*vector_2_norm
    if(denominator == 0): return 0
    return  numerator/denominator

def solve_markov_chain(sentence_to_sentence_graph):
    main_matrix = nx.adjacency_matrix(sentence_to_sentence_graph).toarray()
    W = main_matrix[:-1, :-1]
    last_column = main_matrix[:, -1]
    last_column = last_column[:-1]
    U = np.tile(last_column, (W.shape[0], 1))
    beta = cons.beta
    # Normalize
    U = U / U.sum(axis=1, keepdims=True)
    W = W / W.sum(axis=1, keepdims=True)
    M = beta * U + (1 - beta) * W
    num_rows = M.shape[0]
    P0 = np.ones(num_rows) / num_rows
    
    # Perform Markov chain simulation
    P_history = [P0]
    for k in range(cons.kmax):
        P_k1 = np.dot(M.T, P_history[-1])
        P_history.append(P_k1)

    # Print the results
    for k, P_k in enumerate(P_history):
        print(f"Time step {k}: {P_k}")

    final_probability_vector = P_history[-1]
    print("\nFinal Probability Vector:", final_probability_vector)
    return final_probability_vector

def summarize_and_evaluate(sample_no,test_type,use_ext_only = False):
    sample_path = cons.document_root + "/" + test_type + "/" + sample_no + ".txt"
    input_query = extract_query(sample_path)
    input_doc = extract_doc(sample_path)
    input_summary = extract_summary(sample_path)
    # Get the list of sentences along with their unique ID (0-indexed)
    sentence_with_ids = nlpw.extract_sentences_with_ids(input_doc)
    # Pre-process the sentences. Tokenize, stop word removal
    input_doc.lower()
    processed_sentences_with_ids = nlpw.extract_processed_sentences_with_ids(input_doc)  
    # Pre-process the query
    processed_query = nlpw.process_sentence(input_query)
    # Add query to the main sentences
    max_key = max(processed_sentences_with_ids.keys())
    processed_sentences_with_ids[max_key + 1] = processed_query
    # Create the word set dictionary for all sentences using WordNet
    word_set_dict = nlpw.create_word_set(processed_sentences_with_ids)
    # create the graph to record the score (weight) between every pair of sentences
    sentence_to_sentence_graph = create_s2s_graph(processed_sentences_with_ids,word_set_dict)
    # Solve for the stationary distribution of a markov chain
    sentence_score_vector = solve_markov_chain(sentence_to_sentence_graph)
    # Get the indices of the top K scores                 
    K = max(cons.max_sentences_to_extract,round(0.40*len(processed_sentences_with_ids)))
    if(K >= len(processed_sentences_with_ids)): K = round(0.40*len(processed_sentences_with_ids))
    K = max(K,3) # miniumum length 
    if(use_ext_only): K = 3 # using just QLSK for snippet. No further processing
    top_indices = np.argsort(sentence_score_vector)[-K:]
    # Retrieve the strings from the dictionary based on the top indices
    top_indices.sort() # Mantain sentence order
    top_strings = [sentence_with_ids[i] for i in top_indices] 
    snippet_indices =  np.argsort(sentence_score_vector)[-3:]
    snippet_indices.sort()
    snippet_strings = [sentence_with_ids[i] for i in snippet_indices] 
    snippet_string = ', '.join(snippet_strings)
    # Create a string based on the top strings
    result_string = ', '.join(top_strings)
    return result_string,snippet_string

def use_abs_only(sample_no,test_type):
    sample_path = cons.document_root + "/" + test_type + "/" + sample_no + ".txt"
    input_query = extract_query(sample_path)
    input_doc = extract_doc(sample_path)
    input_summary = extract_summary(sample_path)
    gen_summary = abs2.query_based_summarization_bert(input_doc,input_query)
    print("Summary ---->", gen_summary)
    print("bleu: ", abs2.print_bleu_score(input_summary,gen_summary))

# MAIN switch
#summarize_and_evaluate("16","Informational",False)    # Run test case 16, use only QSLK
#use_abs_only("0","Informational")    # Run test case 0, use only BERT