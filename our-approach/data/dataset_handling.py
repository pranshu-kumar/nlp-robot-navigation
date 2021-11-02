
import pickle
nfile = open('final_dataset.pkl','rb')
f1 = pickle.loads(nfile.read())
print(len(f1))

