from django import forms

class TranslationForm(forms.Form):
    NL_instruction = forms.CharField(
        widget=forms.Textarea(attrs={'class':'uk-textarea'}),
        label='Natural Language Instruction'
        )
