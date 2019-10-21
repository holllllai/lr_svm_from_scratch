import pandas as pd
import numpy as np
INPUT_FILE = '../assg2_release/dating-full.csv'
df = pd.read_csv(INPUT_FILE)
train_out = "trainingSet.csv"
test_out = "testSet.csv"

df = df[:6500]
#I(i)
#assig-2 I(i)

df['race'] = df['race'].str.replace("'", "")
df['race_o'] = df['race_o'].str.replace("'", "")
df['field'] = df['field'].str.replace("'", "")

#assig-2 I(ii)

df['field']=df['field'].str.lower()

#assig-2 I(iv)

list_pref_o = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
list_importance = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important']

df['total_pref_o'] = df[list_pref_o[0]]+ df[list_pref_o[1]]+df[list_pref_o[2]]+df[list_pref_o[3]]+df[list_pref_o[4]]+df[list_pref_o[5]]
for item in list_pref_o:
	df[item] = df[item]/df['total_pref_o']

df['total_importance'] = df[list_importance[0]]+df[list_importance[1]]+df[list_importance[2]]+df[list_importance[3]]+df[list_importance[4]]+df[list_importance[5]]
for item in list_importance:
	df[item] = df[item]/df['total_importance']

df = df.drop(columns=['total_pref_o', 'total_importance'])

#I(ii)
gender_onehot_features = pd.get_dummies(df['gender'])
race_onehot_features = pd.get_dummies(df['race'])
raceo_onehot_features = pd.get_dummies(df['race_o'])
field_onehot_features = pd.get_dummies(df['field'])

gender_dummy_features = gender_onehot_features.iloc[:,:-1]
race_dummy_features = race_onehot_features.iloc[:,:-1]
raceo_dummy_features = raceo_onehot_features.iloc[:,:-1]
field_dummy_features = field_onehot_features.iloc[:,:-1]

def get_dumm_vector(col,value):
	list_ = sorted(df[col].unique())
	lenn = len(list_)-1
	arr = np.zeros(lenn,dtype = int)
	
	for i in range(lenn):
		if list_[i] == value:
			arr[i] = 1

	return arr

print('Mapped vector for female in column gender: ', get_dumm_vector('gender','female'))
print('Mapped vector for Black/African American in column race: ',get_dumm_vector('race','Black/African American'))
print('Mapped vector for Other in column race o: ', get_dumm_vector('race_o','Other'))
print('Mapped vector for economics in column field: ', get_dumm_vector('field', 'economics'))


df = pd.concat([df, gender_dummy_features,race_dummy_features,raceo_dummy_features,field_dummy_features],axis = 1)


df.drop(columns = ['gender','race','race_o','field'], inplace = True)



test=df.sample(frac=0.2,random_state=25) 
train=df.drop(test.index)

train.to_csv(train_out)
test.to_csv(test_out)
