from django.shortcuts import render
from django.http import JsonResponse
from review_app.nlp_model import predict

# Create your views here.

def index(request):
    if request.method == "GET":
        prediction = predict(request.GET.get('review', ''))[0]
        return JsonResponse({'prediction': str(prediction)})
    return JsonResponse({'prediction': -1})
