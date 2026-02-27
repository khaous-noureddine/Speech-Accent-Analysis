import pandas as pd
df=pd.read_pickle("/home/nkhaous/myLLF/speech_accent/representations_distances/representations/mfcc_aligned/layer_1/albanian4_aligned.pkl")
print(df.iloc[450:455]['annotation'])
print(df.columns)

print(df['repr'][0].shape)