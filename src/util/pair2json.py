import json

def pair2json(filename,rfilename):
    file = open(filename,'r',encoding='utf-8')
    lines = file.readlines()
    qid_dict = dict()
    tmp_predicitons = []
    tmp_paraphrases = []
    count = 1
    for line in lines:
        line = line.strip().split('\t')
        print(line)
        if len(tmp_paraphrases)<4:
            tmp_predicitons.append(line[0])
            tmp_paraphrases.append(line[1])
        else:
            qid_dict[count] = (tmp_predicitons,tmp_paraphrases)
            tmp_predicitons = []
            tmp_paraphrases = []
            count += 1

    rfile = open(rfilename,'w',encoding='utf-8')
    json.dump(qid_dict,rfile)

if __name__  == "__main__":
    pair2json('./pipeline_model_restricted_for_BLEU.txt','./pipeline_model_restricted_for_BLEU.json')
