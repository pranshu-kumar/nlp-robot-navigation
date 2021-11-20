from django.shortcuts import render
from .forms import TranslationForm
# Create your views here.
def translate_instruction(request):
    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            return HTTpResponseRedirect('/robottranslation/')
    else:
        form = TranslationForm()

    return render(request, 'translator.html', {'form':form})
