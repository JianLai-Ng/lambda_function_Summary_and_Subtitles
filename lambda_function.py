import json
import urllib
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy
import networkx as nx


import boto3
s3 = boto3.client('s3')


def read_article(body):

    article = body.split(". ")
    #print(article)
    sentences = []

    for sentence in article:
        sentence = sentence.replace("\n","")
        #print(sentence[0])
        sentenced =sentence[0].upper() + sentence[1:]
        sentences.append(sentenced.replace("[^a-zA-Z]", " ").split(". "))
    sentences.pop() 
    
    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = numpy.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    #print("Summarize Text: \n", ". ".join(summarize_text))
    return ". ".join(summarize_text)



def list_summary(text_dict, min_sentences, max_sentences):
    resulting_list ={}
    text_body = text_dict["results"]["transcripts"][0]['transcript']
    for i in range(min_sentences, max_sentences+1):
        resulting_list[i]= generate_summary(text_body, i)
    return resulting_list


def lambda_handler(event, context):

    bucket_1 = event['Records'][0]
    bucket_2 = bucket_1['s3']['bucket']['name']
   
    key_1 = event['Records'][0]
    key_2 = urllib.parse.unquote_plus(key_1['s3']['object']['key'], encoding = 'utf-8')
    
    response = s3.get_object(Bucket = bucket_2, Key = key_2)
    
    #Load the json file
    text = response['Body'].read().decode()
    data = json.loads(text)
    
    min_sent = 1 #CHANGE
    max_sent = 3 #CHANGE
    
    result_summary = list_summary(data, min_sent,max_sent)
    key_name = "summaryjson_" + (".".join(data["jobName"].split(".")[1:][:-1])) + ".json"
    print("result_summary")
    print(result_summary)
    print("saved as")
    print(key_name)
    

    s3.put_object(Bucket="cs5224-text-summary", Key=key_name, Body=result_summary)
