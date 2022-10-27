from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect
from sklearn.pipeline import Pipeline
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
# from .doc_classify import class_predict

from .pipeline_Extract import Extraction
from .pipeline_Transform import Transformation
from .pipeline_Csv_Json import Csv_json
from .pipeline_Text_Pdf import Text_PDF
from .pipeline_Json_Text import json_text

from .models import allmodel
from django.contrib import messages

# steps = [('scaler', StandardScaler()), ('SVM', SVC())]
# from sklearn.pipeline import Pipeline
# pipeline = Pipeline(steps)

def Extractor(request):
    types = list(allmodel.objects.values_list('Modelname', flat=True))
    print(types)
    print(len(types))
    no_model = list(range(1,len(types)+1))
    print(no_model)
    if request.method == "POST":
        request_file = request.FILES.get('Document')
        print(request_file)
        fs = FileSystemStorage()
        file = fs.save(request_file.name, request_file)
        pdf_doc = "D:\\scb_bazaar\\scb_bazaar\\scb_dynamic_dropdown\\scb_project\\media\\" + file
        model = request.POST.getlist('model')
        print("model.......", model)
        if '+' in model:
            model.remove('+')
        print("model.......", model)
        list1 = []
        dict_function = {}
        for i in model:
            list1.append(globals()[i])
            dict_function[i]=globals()[i]

        print(list1)
        print(dict_function)
        steps = [(k, v) for k, v in dict_function.items()]
        print(steps)

        steps = []
        print("***********",pdf_doc)
        print("+++++++++++",list1[0](pdf_doc))
        a1 = list1[0](pdf_doc)
        for i in range(1,len(list1)):
            print(i)
            a1 = list1[i](a1)
            print(a1)
        #pipeline= list1[0](pdf_doc)
        # for i in range(1,len(list1)):
        #     print(i)
        #     pipeline = Pipeline(steps=steps=[('add',list1[i](pipeline))])

        # print("++++++++++++++",pipeline)
        # pipeline=Pipeline(steps=steps)
        # print("--------------",pipeline)




    if request.method == "GET":
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            print("AJAX CALLED POST")
            Value = request.GET.get('data')
            print('Value : ', Value)
            queryset = allmodel.objects.filter(Modelname = Value).values('Output')
            # User.objects.filter(age__isnull=True).values('id','age')
            print(queryset)
            print(queryset[0]['Output'])
            model_output = queryset[0]['Output']
            queryset_1 = allmodel.objects.filter(Input = model_output).values_list('Modelname', flat=True)
            print(queryset_1)
            html = ""
            for value in queryset_1:
                print(value) 
                html += "<option value=" +value+">" +value+ "</option>"
            return HttpResponse(html)

    return render(request, "index.html",{'form': types,'No_model': no_model})
#
#
def transform(request):
    return render(request, "transform.html")

def classification(request):
    print("Hi")
    return render(request, "classification.html")



