from django import forms

class TranslationForm(forms.Form):
    nl_instruction = forms.CharField(
        widget=forms.Textarea(attrs={'class':'uk-textarea'}),
        label='Natural Language Instruction'
        )
    b_map = forms.CharField(
        widget=forms.Textarea(attrs={'class':'uk-textarea'}),
        label='Behaviour Map'
        )

    start_node = forms.CharField(
        widget=forms.TextInput(attrs={'class':'uk-input'}),
        label='Start Node'
    )
