import json
import urllib
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy
import networkx as nx
from botocore.errorfactory import ClientError


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


################################################################## SUBTITLE FUNC ##################################################################
def subs_list_maker(job, len_option1 = 10 , len_option2 = 12):
    
    
    
###### Correct 1st sentence 1st word to capital letter   
    list1 = job['results']['transcripts'][0]['transcript'].split(". ")
    list1[0] = list1[0][0].upper() + list1[0][1:]
    list1
###### Add full stops where appropriate
    list1 = [x.strip() for x in list1]
    list1 = [x+'.' if x[-1]!='.'else x for x in list1]
    list1
###### Split sentences to tokens fo appropriate length
    sentence_tokens = []
    dabao = False
    dabao_item = ''


    for sent in list1:
        sent = sent.split(" ")


        numwords = len(sent)
        if numwords <= 3:
            dabao = True
            dabao_item = sent
            continue



        if dabao == True:
            sent = dabao_item + sent

        numwords = len(sent)
        #print(numwords)
        cutter_num = (len_option2 if numwords%len_option2 == 0 else len_option1 if (numwords%len_option1 > numwords%len_option2) else len_option2)
        #print(cutter_num)
        word_tokens_sentence = sent
        list_sent_tokens = [word_tokens_sentence[i:i+cutter_num] for i in range(0, len(word_tokens_sentence), cutter_num)]
        for sent_token in list_sent_tokens:

            if len(sent_token) <= 3:
                dabao = True
                dabao_item = sent_token
                continue
            else:
                dabao = False
                dabao_item = ''
            #print(" ".join(sent_token))
            sentence_tokens.append(sent_token)

            #print("")



###### Split sentences to tokens of appropriate length
    word_time = []

    for eachword_dict in job['results']['items']:
        try:
            start_time = eachword_dict['start_time']
            end_time = eachword_dict['end_time']
            this_word = eachword_dict['alternatives'][0]['content']
            word_time.append([this_word, start_time, end_time])
        except:
            pass

###### Get start and end timings of each sentence tokens
    counter  =0
    next_counter = 0
    result_puzzle = []
    for sentok in sentence_tokens:
        #['Machine','learning','is','employed','in','a','range','of','computing','tasks','where','designing']
        if counter !=0:
            counter+=1
        sent_start = word_time[counter][1]

        counter= counter+len(sentok)-1
        sent_end = word_time[counter][2]
        result_puzzle.append([sent_start, sent_end, sentok])


###### Create list to store entries of srt output file
    counter = 1
    srt_list=[]
    for aset in result_puzzle:
        start_time = aset[0]
        end_time = aset[1]
        sentence = " ".join(aset[2])
        if counter!=1:
            srt_list.append("\n")

        start_interm = '{0:.10f}'.format(float(start_time)/1000000).split(".")[-1]
        start_timing = start_interm[:2] + ":" + start_interm[2:4] + ":" + start_interm[4:6] + "," + start_interm[6:8] 

        end_interm = '{0:.10f}'.format(float(end_time)/1000000).split(".")[-1]
        end_timing = end_interm[:2] + ":" + end_interm[2:4] + ":" + end_interm[4:6] + "," + end_interm[6:8] 

        timing = start_timing + " --> " + end_timing



        srt_list.append(str(counter)+'\n')
        srt_list.append(str(timing)+'\n')
        srt_list.append(str(sentence)+'\n')
        counter+=1
        
    return srt_list
####################################################################################################################################

################################################################## SUMMARY FUNC ##################################################################
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

def create_dict_from_tagset(tagset):
    op_dict = {}
    for tagpair in tagset:
        key = tagpair['Key']
        val = tagpair['Value']
        op_dict[key] = val
    return op_dict

####################################################################################################################################

def lambda_handler(event, context):

    bucket_1 = event['Records'][0]
    bucket_2 = bucket_1['s3']['bucket']['name']
   
    key_1 = event['Records'][0]
    key_2 = urllib.parse.unquote_plus(key_1['s3']['object']['key'], encoding = 'utf-8')
    
    response = s3.get_object(Bucket = bucket_2, Key = key_2)
    
    print("got response")
    
    #Load the transcribed json file
    text = response['Body'].read().decode()
    data = json.loads(text)
    
    print("loaded text")    
    
    #Load the tag set
    
    op_dict = {}
    while len(response_dict)==0:
        try:
            # TODO: write code...
            op_dict = s3.get_object_tagging(Bucket='cs5224-text', Key=job_name + '.json')
            print('Tag Detected in Bucket')
        except ClientError:
            time.sleep(20)
            print('Retrying')
            continue

    
  
    
    tag_dict = create_dict_from_tagset(op_dict['TagSet'])
    min_sent = int(tag_dict['summary_option_1'])
    max_sent = int(tag_dict['summary_option_2'])
    type_process = tag_dict['summary_type']

    print("parsed requirement")
    print(str(type_process) + str(min_sent)+"_" +str(max_sent))
    
    if type_process == 'Summary':
        print("creating summary")
        result_summary = list_summary(data, min_sent,max_sent)
        key_name = ".".join(data["jobName"].split(".")[:-1]) + ".json"
        print("result_summary")
        
        print(result_summary)
        print("saved as")
        print(key_name)

        s3.put_object(Bucket="cs5224-text-summary", Key=key_name, Body=json.dumps(result_summary).encode())
        
    elif type_process == 'Subtitle':
        print("creating subtitles")
        subbed_list = subs_list_maker(data)
        key_name = ".".join(data["jobName"].split(".")[:-1]) + ".srt"

        print("Subtitles")
        print("saved as")
        print(key_name)


        s3.put_object(Bucket="cs5224-subtitles-output", Key=key_name, Body=subbed_list)        
        
        
        
    else:
        print('Non existent key to define process')
        
        
