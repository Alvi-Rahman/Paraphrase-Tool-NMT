from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from django.conf import settings
import torch
import os
from django.views import View
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
import random


class ParaphraseTool:
    def __init__(self):
        self.set_seed(42)
        self.__model, self.__tokenizer, self.__device = self.load_model()
    
    def get_model(self):
        return self.__model
    
    def get_tokenizer(self):
        return self.__tokenizer

    def get_device(self):
        return self.__device

    def set_seed(self,seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def load_model(self):
        model = PegasusForConditionalGeneration.from_pretrained(os.path.join(
            settings.BASE_DIR, 'paraphrase_utils', 'model'))
        tokenizer = PegasusTokenizer.from_pretrained(os.path.join(
            settings.BASE_DIR, 'paraphrase_utils', 'tokenizer'))   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return model, tokenizer, device

    def paraphrase(self, sentence):
        context = sent_tokenize(sentence)
        if len(context[-1]) < 5:
            context = context[:-1]

        encoding = self.__tokenizer.prepare_seq2seq_batch(context, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(
                        self.__device), encoding["attention_mask"].to(self.__device)

        translated = self.__model.generate(input_ids=input_ids,
                                                attention_mask=attention_masks,
                                                do_sample=True,
                                                min_length=max([len(i.split()) for i in context]) - 5,
                                                max_length=max([len(i.split()) for i in context]) + 5,
                                                top_k=120,
                                                top_p=0.95,
                                                temperature=0.98,
                                                early_stopping=True,
                                                num_return_sequences=1,
                                                no_repeat_ngram_size=3
                                            )
        tgt_text = self.__tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        final = ' '.join(tgt_text)
        tokens = nltk.word_tokenize(final)
        tag = nltk.pos_tag(tokens)
        tags_only = [i[1] for i in tag]
        selected = [i for i, x in enumerate(tags_only) if x == "NN" or x == "JJ"]
        syn = []
        for i in selected:
            try:
                syn.append(wordnet.synsets(tokens[i])[-1].lemmas()[-1].name())
            except:
                syn.append(tokens[i])
                pass

        j = 0
        for i in selected:
            tokens[i] = syn[j]
            j += 1

        final = ' '.join(tokens)

        return final

class HomePage(View):
    template_name = "paraphrase_app/index.html"
    paraphrase_class = ParaphraseTool()

    def get(self, request):
        # paraphrase_class = ParaphraseTool()
        return render(request, "paraphrase_app/index.html", context={'get':1})

    def post(self, request):

        real_text = request.POST['paraphrase_content']
        if len(real_text.split()) < 5:
            paraphrased_data = None
        else:
            paraphrased_data = self.paraphrase_class.paraphrase(real_text)
        context = {
            'real_text': real_text,
            'paraphrased_data': paraphrased_data,
        }
        return render(request, "paraphrase_app/index.html", context=context)

# def home_page(request):
#     if request.method == 'GET':
#         return render(request, "paraphrase_app/index.html")
#     elif request.method == 'POST':
#         real_text = request.POST['paraphrase_content']
#         if len(real_text) < 10:
#             paraphrased_data = None
#         else:
#             paraphrased_data = os.listdir(os.path.join(
#                 settings.BASE_DIR, 'paraphrase_utils', 'model'))
#         context = {
#             'real_text': real_text,
#             'paraphrased_data': paraphrased_data,
#         }
#         return render(request, "paraphrase_app/index.html", context=context)

