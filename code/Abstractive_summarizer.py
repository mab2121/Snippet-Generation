from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from summarizer import Summarizer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

pega_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
pega_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

def query_based_summarization_t5(input_text, query, max_length=70, model_name="t5-base"): #150
   
    # Combine query with input text for query-based summarization
    input_text_with_query = ""
    if(query is None): input_text_with_query = input_text
    else: input_text_with_query = f"summarize: {query} - {input_text}"

    # Tokenize and generate summary
    input_ids = t5_tokenizer.encode(input_text_with_query, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = t5_model.generate(input_ids, max_length=max_length, length_penalty=2.0, num_beams=6, early_stopping=True)

    # Decode the summary
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_text_bart(input_text, query, max_length=70, model_name="facebook/bart-large-cnn"):
    # Load BART tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Combine query with input text for query-based summarization
    input_text_with_query = ""
    if(query is None): input_text_with_query = input_text
    else: input_text_with_query = f"summarize: {query} - {input_text}"

    # Tokenize and generate summary
    input_ids = bart_tokenizer.encode(input_text_with_query, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = bart_model.generate(input_ids, max_length=max_length, length_penalty=2.0, num_beams=6, early_stopping=True)

    # Decode the summary
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_text_pegasus(input_text, query, model_name="google/pegasus-xsum", max_length=70):
    # Load PEGASUS tokenizer and model
    # Combine query with input text for query-based summarization
    input_text_with_query = ""
    if(query is None): input_text_with_query = input_text
    else: input_text_with_query = f"summarize: {query} - {input_text}"

    # Tokenize and generate summary
    input_ids = pega_tokenizer.encode(input_text_with_query, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = pega_model.generate(input_ids, max_length=max_length, length_penalty=2.0, num_beams=6, early_stopping=True)

    # Decode the summary
    summary = pega_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def query_based_summarization_bert(input_text, query,num_sentences=2):
    model = Summarizer()

    # Combine query with input text for query-based summarization
    input_text_with_query = f"{query} {input_text}"

    # Generate summary
    summary = model(input_text_with_query,num_sentences=num_sentences)

    return summary

def print_bleu_score(reference_summary,generated_summary):
    reference_tokens = reference_summary.split()
    hypothesis_tokens = generated_summary.split()
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=SmoothingFunction().method1)
    return bleu_score

