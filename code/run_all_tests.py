import Constants as cons
import QLSK as qlsk
import Abstractive_summarizer as abs2
import numpy as np

def run_all_summarization_tests(test_type):
    inputs = 40
    file_path = cons.test_results_file
    qlsk_sum = []
    bert_sum = []
    t5_sum = []
    fb_sum = []
    pega_sum = []
    t5_qlsk_sum = []
    fb_qlsk_sum = []
    pega_qlsk_sum = []
    best_qlsk_hyb_sum = []
    win_counts = [0] * 8
    with open(file_path, 'a') as file:
        for i in range(0,41):
            print("======================> TEST NO:",i)
            file.write("-"*150 + "\n")
            file.write("TEST NO: "+ str(i) + "\n")
            file.write("-"*150 + "\n")
            sample_path = cons.document_root + "/" + test_type + "/" + str(i) + ".txt"
            input_query = qlsk.extract_query(sample_path)
            input_doc = qlsk.extract_doc(sample_path)
            input_summary = qlsk.extract_summary(sample_path)
            file.write("Google Snippet: "  + input_summary + "\n")
            qlsk_only_result, qlsk_only_snippet = qlsk.summarize_and_evaluate(str(i),"Informational",False)
            file.write("---------- QLSK Only ----------------\n")
            file.write("Snippet: " + qlsk_only_snippet  + "\n")
            bleu1 = abs2.print_bleu_score(input_summary,qlsk_only_snippet)
            file.write("Bleu: " + str(bleu1) + "\n")
            qlsk_sum.append(bleu1)
            file.write("---------- Bert Only (Query Guided) ----------------\n")
            bert_only_result = abs2.query_based_summarization_bert(input_doc,input_query)
            file.write("Snippet: " + bert_only_result + "\n")
            bleu2 = abs2.print_bleu_score(input_summary,bert_only_result)
            file.write("Bleu: " + str(bleu2) + "\n")
            bert_sum.append(bleu2)
            file.write("---------- T5 Only (Query Guided) ----------------\n")
            t5_only_result = abs2.query_based_summarization_t5(input_doc,input_query)
            file.write("Snippet: " + t5_only_result + "\n")
            bleu3 = abs2.print_bleu_score(input_summary,t5_only_result)
            file.write("Bleu: " + str(bleu3) + "\n")
            t5_sum.append(bleu3)
            file.write("---------- FB Only (Query Guided) ----------------\n")
            fb_only_result = abs2.summarize_text_bart(input_doc,input_query)
            file.write("Snippet: " + fb_only_result + "\n")
            bleu4 = abs2.print_bleu_score(input_summary,fb_only_result)
            file.write("Bleu: " + str(bleu4) + "\n")
            fb_sum.append(bleu4)
            file.write("---------- T5 + QLSK ----------------\n")
            t5_qlsk_result = abs2.query_based_summarization_t5(qlsk_only_result,None)
            file.write("Snippet: " + t5_qlsk_result + "\n")
            bleu5 = abs2.print_bleu_score(input_summary,t5_qlsk_result)
            file.write("Bleu: " + str(bleu5) + "\n")
            c1 = bleu5
            t5_qlsk_sum.append(bleu5)
            file.write("---------- FB + QLSK----------------\n")
            fb_qlsk_result = abs2.summarize_text_bart(qlsk_only_result,None)
            file.write("Snippet: " + fb_qlsk_result + "\n")
            bleu6 = abs2.print_bleu_score(input_summary,fb_qlsk_result)
            c2 = bleu6
            file.write("Bleu: " + str(bleu6) + "\n")
            fb_qlsk_sum.append(bleu6)
            file.write("---------- Pega + QLSK ----------------\n")
            pega_qlsk_result = abs2.summarize_text_pegasus(qlsk_only_result,None)
            file.write("Snippet: " + pega_qlsk_result + "\n")
            bleu7 = abs2.print_bleu_score(input_summary,pega_qlsk_result)
            c3 = bleu7
            file.write("Bleu: " + str(bleu7) + "\n")
            pega_qlsk_sum.append(bleu7)
            bleu8 = max(max(c1,c2),c3)
            best_qlsk_hyb_sum.append(bleu8)
            scores = [bleu1, bleu2, bleu3, bleu4, bleu5, bleu6, bleu7,bleu8]
            winner_index = scores.index(max(scores))
            win_counts[winner_index] += 1
    with open(file_path, 'a') as file:        
        file.write("="*150)
        file.write("\n\n\n")
        file.write("Final Results: \n\n")
        file.write(f"1. QLSK Only: Mean: {np.mean(qlsk_sum)}, Max: {np.max(qlsk_sum)}, Median: {np.median(qlsk_sum)}, Min: {np.min(qlsk_sum)}, 25th Percentile: {np.percentile(qlsk_sum, 25)}, 75th Percentile: {np.percentile(qlsk_sum, 75)}\n")
        file.write(f"2. Bert Only: Mean: {np.mean(bert_sum)}, Max: {np.max(bert_sum)}, Median: {np.median(bert_sum)}, Min: {np.min(bert_sum)}, 25th Percentile: {np.percentile(bert_sum, 25)}, 75th Percentile: {np.percentile(bert_sum, 75)}\n")
        file.write(f"3. T5 Only: Mean: {np.mean(t5_sum)}, Max: {np.max(t5_sum)}, Median: {np.median(t5_sum)}, Min: {np.min(t5_sum)}, 25th Percentile: {np.percentile(t5_sum, 25)}, 75th Percentile: {np.percentile(t5_sum, 75)}\n")
        file.write(f"4. FB Only: Mean: {np.mean(fb_sum)}, Max: {np.max(fb_sum)}, Median: {np.median(fb_sum)}, Min: {np.min(fb_sum)}, 25th Percentile: {np.percentile(fb_sum, 25)}, 75th Percentile: {np.percentile(fb_sum, 75)}\n")
        file.write(f"5. T5 QLSK: Mean: {np.mean(t5_qlsk_sum)}, Max: {np.max(t5_qlsk_sum)}, Median: {np.median(t5_qlsk_sum)}, Min: {np.min(t5_qlsk_sum)}, 25th Percentile: {np.percentile(t5_qlsk_sum, 25)}, 75th Percentile: {np.percentile(t5_qlsk_sum, 75)}\n")
        file.write(f"6. FB QLSK: Mean: {np.mean(fb_qlsk_sum)}, Max: {np.max(fb_qlsk_sum)}, Median: {np.median(fb_qlsk_sum)}, Min: {np.min(fb_qlsk_sum)}, 25th Percentile: {np.percentile(fb_qlsk_sum, 25)}, 75th Percentile: {np.percentile(fb_qlsk_sum, 75)}\n")
        file.write(f"7. Pega QLSK: Mean: {np.mean(pega_qlsk_sum)}, Max: {np.max(pega_qlsk_sum)}, Median: {np.median(pega_qlsk_sum)}, Min: {np.min(pega_qlsk_sum)}, 25th Percentile: {np.percentile(pega_qlsk_sum, 25)}, 75th Percentile: {np.percentile(pega_qlsk_sum, 75)}\n")
        file.write(f"8. BEST QLSK HYBRID: Mean: {np.mean(best_qlsk_hyb_sum)}, Max: {np.max(best_qlsk_hyb_sum)}, Median: {np.median(best_qlsk_hyb_sum)}, Min: {np.min(best_qlsk_hyb_sum)}, 25th Percentile: {np.percentile(best_qlsk_hyb_sum, 25)}, 75th Percentile: {np.percentile(best_qlsk_hyb_sum, 75)}\n")
        file.write("\n".join(f"Model {i + 1}: {count}" for i, count in enumerate(win_counts)))
        file.write("\n\n\n")
# Main switch
run_all_summarization_tests("Informational") # Run all tests