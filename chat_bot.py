#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load the Packages
from rasa_nlu.training_data  import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Metadata, Interpreter
import random
import re


# In[2]:


# Loading DataSet
train_data = load_data('rasa_dataset.json')
# Config Backend using Sklearn and Spacy
trainer = Trainer(config.load("config_spacy.yaml"))
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)


# In[3]:


# Training Data
trainer.train(train_data)


# In[4]:


# g = "I am looking for an Italian Restaurant where I can eat"
def classify(sentence):
    model_directory = trainer.persist('projects/')
    interpreter = Interpreter.load(model_directory)
    t = interpreter.parse(str(sentence))
    result = [(str(t['intent']['name']),t['intent']['confidence'])]
    print(result)
    return result


# In[129]:


context = {}
def responsee(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']


                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or                         (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details:
                            print ('tag:', i['tag'])
                        # a random response from the intent
                        return str(random.choice(i['responses']))

            results.pop(0)
            return "sorry i don't understand what you say please call our administration team they will help you "


# In[130]:


def chat(a):
    r = re.compile("(hello|hey|hi) my name is (.+)")
    match = r.search(a.lower())
    if a == 'stop':
        pass
    if match:
        name = match.groups()[1]
        a = "hi"
        b = "welcome to mabrotrix"
        response = a + " " + name + " " + b
    else:
        response = responsee(a)
        print(response)
    return str(response)


# In[132]:


# a = input()
# response(a)


# In[ ]:




