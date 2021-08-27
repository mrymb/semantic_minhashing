from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
import random as rand
import pandas as pd
from nltk import ngrams
import re
from sklearn.metrics import accuracy_score

class minhash: #returns for all

    def __init__(self,str1=None,str2=None,k=5,hops=None):
        
        self.str1=str1
        self.str2=str2
        self.k=k     
        self.codes_dict={}
        self.counter=1
     

        if str1 and str2:
         
            self.vectorize(hops)
            self.generate_hashfuncs()

    def stringtokenizer(self,text):

        tokenizer = RegexpTokenizer(r"\w+")  #remove punctuations
        tokens = tokenizer.tokenize(text.lower())
        return tokens

    def remove_stopwords(self,text):

        tokens = self.stringtokenizer(text)
        tokens_without_sw = [word for word in tokens if not word in stopwords.words()]
        return tokens_without_sw

        
    def get_codes(self,tokens):
    
    
        v=[]
        for word in tokens:
        
            if self.codes_dict and word in self.codes_dict.keys():
                v.append(self.codes_dict[word])
                
            else:
                self.codes_dict[word]=self.counter
                v.append(self.counter)
                self.counter+=1
                
        return v  

    def hash(self,x,a,b,p):     #(x,a,b,p,N):
        return (a*x + b) % p    #(((a*x)+b) % p)%N

    
    def generate_coefficients(self,k,maxID):

        coefficients=[] 
        for i in range(k):
            a=rand.randint(1,maxID)
            b=rand.randint(0,maxID)
        
            coefficients.append((a,b))
        
        return coefficients

    def nextPrime(self,n):

        l=[*range(n+1,2*n)]
        return min([i for i in l if all([i % n for n in range(2,i)])])

    def get_ngrams(self,docx,n=3):

        clean_doc=re.sub(r'[^\w\s]', '',docx.lower())
        doc=ngrams(clean_doc.split(),n)
    
        grams=[]
        for d in doc:
            
            g=' '.join(list(d))
            grams.append(g)
        
        return grams

    def compute_signature(self,v):
    
        signature=[]
     
        for i in range(self.k):
            minHashCode = self.nextprime +1
            for c in set(v):
                a,b=self.coef[i]
                h=self.hash(c,a,b,self.nextprime)      # The problem
            
                if h<minHashCode:
                    minHashCode=h
            signature.append(minHashCode)
            
        return signature

    
    def jaccard_similarity(self):

        intersection = len(list(set(self.cv1).intersection(self.cv2)))
        union = (len(self.cv1) + len(self.cv2)) - intersection
        self.jaccard_similarity_score=float(intersection) / union
        return self.jaccard_similarity_score
    
    def compute_minhash_similarity(self,c1,c2):

        comparisons=[]
    
        for i in range(self.k):

            if c1[i]==c2[i]:
                comparisons.append(1)
            else:
                comparisons.append(0)
    
        min_sim=sum(comparisons)/self.k  
        return min_sim
    
    def compute_lexical_minhash_similarity(self):

        
        sig1=self.compute_signature(self.cv1)
        sig2=self.compute_signature(self.cv2)
    
        self.minhash_lexical_similarity_score= self.compute_minhash_similarity(sig1,sig2)
        return self.minhash_lexical_similarity_score


    def display(self):
    
        print('String 1: ',self.str1)
        print('String 2:',self.str2)
        print('Jaccard Similarity: ',self.jaccard_similarity())
        print('Minhash Lexical Similarity: ',self.compute_lexical_minhash_similarity())
        print('Minhash Semantic Similarity: ',self.compute_semantic_minhash_similarity_synonyms())
        print('Minhash Semantic Similarity: ',self.compute_semantic_minhash_similarity_antonyms())
        print('Minhash Semantic Similarity: ',self.compute_semantic_minhash_similarity_hyponyms())
        print('Minhash Semantic Similarity: ',self.compute_semantic_minhash_similarity_hypernyms())
           

    def get_synonyms(self,word):
    
        synonyms = [] 

        for syn in wn.synsets(word):      
        
            for l in syn.lemmas(): 
                synonyms.append(l.name()) 
              
        return list(set(synonyms)) 

    def get_antonyms(self,word,thresh=3):
    
        antonyms = []
    
        for syn in wn.synsets(word):
            for l in syn.lemmas():
            
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        
        if thresh:
            return list(set(antonyms))[:thresh]
        else:
            return list(set(antonyms))

    def get_hyponyms(self,word,thresh=3):
    
        hyponyms=[]
        syn_hyponyms=wn.synsets(word)

        if syn_hyponyms:
            for hyp in syn_hyponyms[0].hyponyms():
                hyponyms.append((hyp.lemmas()[0].name()))
            
            if thresh and len(hyponyms)>thresh:
                return hyponyms[:thresh]
            else:
                return hyponyms
        else:
            return hyponyms

    # def get_hypernyms(self,word,thresh=3):
    
    #     hypernyms=[]
    #     syn_hyper=wn.synsets(word)
    #     if syn_hyper:
    #         for hyp in syn_hyper[0].hypernyms():
    #             hypernyms.append((hyp.lemmas()[0].name()))
            
    #         if thresh and len(hypernyms)>thresh:
    #             return hypernyms[:thresh]
    #         else:
    #             return hypernyms
    #     else:
    #         return hypernyms

    def next_hypernyms(self,synsets):
        hypernyms=[]    
        for h in synsets:       
            hypernyms+=h.hypernyms()    
        return hypernyms

    def get_hypernyms(self,word, hops=1, thresh=3):
    
        start=wn.synsets(word)
        if start:
            next_level=start[0].hypernyms()
    
            for i in range(hops): 
                hypernyms=[]
                for h in next_level:
                    hypernyms.append(h)
            
        
                next_level=self.next_hypernyms(next_level)
     
            hyp=[]
            for h in hypernyms:
                name=h.name()
                hyp.append(name.split('.')[0])

            if thresh and len(hyp)>thresh:
                return hyp[:thresh]
            else:
                return hyp

        else:
            return []



    def get_extended_vectors(self,doc,threshold=3,hop=1,choice='s'):
    
        ev=[]
        words=self.remove_stopwords(doc)
        
        #variables for collecting stats
        self.numSynonyms=0
        self.numAntonyms=0
        self.numHyponyms=0
        self.numHypernyms=0

       # lemmatizer=WordNetLemmatizer()
        syns=[]
        for word in words:
            
           # word=lemmatizer.lemmatize(word)

            if choice=='s':
                syns=self.get_synonyms(word)
                self.numSynonyms+=len(syns)
            elif choice=='a':
                syns=self.get_antonyms(word)
                self.numAntonyms+=len(syns)
            elif choice=='hypo':
                syns=self.get_hyponyms(word)
                self.numHyponyms+=len(syns)
            elif choice=='hyper':
                syns=self.get_hypernyms(word,hops=hop)
                self.numHypernyms+=len(syns)

            if syns and len(syns)>=threshold:
                ev.append(word)
                ev+=syns[:threshold]
            else:
                ev.append(word)
                ev+=syns

        return ev

    def vectorize(self,hops=None):

        #for lexical similarity
        gram1=self.get_ngrams(self.str1,n=3)
        gram2=self.get_ngrams(self.str2,n=3)
        
        self.cv1=self.get_codes(gram1)
        self.cv2=self.get_codes(gram2)

        #for semantic synonyms
        self.cev1=self.get_codes(self.get_extended_vectors(self.str1))
        self.cev2=self.get_codes(self.get_extended_vectors(self.str2))

        #for semantic antonyms
        self.cev1a=self.get_codes(self.get_extended_vectors(self.str1,choice='a'))
        self.cev2a=self.get_codes(self.get_extended_vectors(self.str2,choice='a'))

        #for semantic hyponyms
        self.cev1hypo=self.get_codes(self.get_extended_vectors(self.str1,choice='hypo'))
        self.cev2hypo=self.get_codes(self.get_extended_vectors(self.str2,choice='hypo'))

        #for semantic hypernyms
        if hops:
            self.cev1hyper=self.get_codes(self.get_extended_vectors(self.str1,choice='hyper',hop=hops))
            self.cev2hyper=self.get_codes(self.get_extended_vectors(self.str2,choice='hyper',hop=hops))
        else: 
            self.cev1hyper=self.get_codes(self.get_extended_vectors(self.str1,choice='hyper'))
            self.cev2hyper=self.get_codes(self.get_extended_vectors(self.str1,choice='hyper'))


    def generate_hashfuncs(self):

        codes=self.cv1+self.cv2+self.cev1+self.cv2
        self.maxID=max(codes)
        self.coef=self.generate_coefficients(self.k,self.maxID)
        self.nextprime=self.nextPrime(self.maxID)

    def compute_semantic_minhash_similarity_synonyms(self):
        
        sig1=self.compute_signature(self.cev1)
        sig2=self.compute_signature(self.cev2)
      
        self.minhash_semantic_similarity_score= self.compute_minhash_similarity(sig1,sig2)
        return self.minhash_semantic_similarity_score

    def compute_semantic_minhash_similarity_antonyms(self):
        sig1=self.compute_signature(self.cev1a)
        sig2=self.compute_signature(self.cev2a)

        self.minhash_semantic_similarity_score_ant=self.compute_minhash_similarity(sig1, sig2)
        return self.minhash_semantic_similarity_score_ant


    def compute_semantic_minhash_similarity_hyponyms(self):
        sig1=self.compute_signature(self.cev1hypo)
        sig2=self.compute_signature(self.cev2hypo)

        self.minhash_semantic_similarity_score_hypo=self.compute_minhash_similarity(sig1, sig2)
        return self.minhash_semantic_similarity_score_hypo

    def compute_semantic_minhash_similarity_hypernyms(self):
        sig1=self.compute_signature(self.cev1hyper)
        sig2=self.compute_signature(self.cev2hyper)

        self.minhash_semantic_similarity_score_hyper=self.compute_minhash_similarity(sig1, sig2)
        return self.minhash_semantic_similarity_score_hyper

    
    def compute(self):
        self.jaccard_similarity()
        self.compute_lexical_minhash_similarity()
        self.compute_semantic_minhash_similarity_synonyms()
        self.compute_semantic_minhash_similarity_antonyms()
        self.compute_semantic_minhash_similarity_hyponyms()
        self.compute_semantic_minhash_similarity_hypernyms()



def main():
    
    

    # with open("semantic_data.csv",encoding='utf8') as file:
    #     data=file.read()
    
    # lines=data.split('\n')

    # records=[]
    # for line in lines:
    #     records.append(line.split('\t'))


    # df=pd.DataFrame(records,columns=['Quality','ID1','ID2','str1','str2']).dropna().drop(['ID1','ID2'],axis=1)
    df=pd.read_csv('SICK.txt',sep='\t',usecols=['sentence_A','sentence_B','relatedness_score' ])
    df.columns=['str1','str2','relatedness_score']
    
    # df['Jaccard']=0
    # df['lexical']=0
    # df['synonyms']=0
    # df['antonyms']=0
    # df['hyponyms']=0
    # df['hypernyms']=0
    # n=len(df)
    # c=n-1
    # for i in range(1,n):
    #     print('Processing Sentence: ',i)
    #     m=minhash2(df.str1.iloc[i],df.str2.iloc[i],k=7)
    #     m.compute()
    #     c-=1
    
    #     df.loc[i,'Jaccard']=m.jaccard_similarity_score
    #     df.loc[i,'lexical']=m.minhash_lexical_similarity_score
    #     df.loc[i,'synonyms']=m.minhash_semantic_similarity_score
    #     df.loc[i,'antonyms']=m.minhash_semantic_similarity_score_ant
    #     df.loc[i,'hyponyms']=m.minhash_semantic_similarity_score_hypo
    #     df.loc[i,'hypernyms']=m.minhash_semantic_similarity_score_hyper

    df['hypernyms_1']=0
    df['hypernyms_2']=0
    df['hypernyms_3']=0

    n=len(df)
 
    for i in range(1,n):
        print('Processing Sentence: ',i)
        m=minhash(df.str1.iloc[i],df.str2.iloc[i],k=7,hops=1)
        m.compute()
        df.loc[i,'hypernyms_1']=m.minhash_semantic_similarity_score_hyper

        m=minhash2(df.str1.iloc[i],df.str2.iloc[i],k=7,hops=2)
        m.compute()
        df.loc[i,'hypernyms_2']=m.minhash_semantic_similarity_score_hyper

        m=minhash2(df.str1.iloc[i],df.str2.iloc[i],k=7,hops=3)
        m.compute()
        df.loc[i,'hypernyms_3']=m.minhash_semantic_similarity_score_hyper
    



    df=df.drop(['str1','str2'],axis=1)
    #df=df.drop(['sentence_A','sentence_B'],axis=1)
    df.to_csv('results2_hypernyms_sick.csv',index=False)
  

if __name__ == '__main__':
    main()