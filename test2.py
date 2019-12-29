import pandas as pd
from textblob import TextBlob

data = pd.read_csv("tweets_turkcell_end.csv", delimiter="•", encoding='utf-8')
# Preview the first 5 lines of the loaded data
# data.fillna("")
# texts = []
# for index, row in data.iterrows():
#     text = row['date'] + "•" + row['tweet']
#     if not pd.isnull(row['tweet2']):
#         text = text + " " + row['tweet2']
#         if not pd.isnull(row['tweet3']):
#             text = text + " " + row['tweet3']
#             if not pd.isnull(row['tweet4']):
#                 text = text + " " + row['tweet4']
#                 if not pd.isnull(row['tweet5']):
#                     text = text + " " + row['tweet5']
#     texts.append(text)
#
# f = open('tweets_turkcell_end.csv', 'w')
# for line in texts:
#     f.write(line + '\n')  # Give your csv text here.
# f.close()
dataResult = pd.read_csv("tweets_turkcell_result.csv", delimiter="•", encoding='utf-8')
data['RESULT'] = dataResult

data.to_csv("tweets_turkcell_result.csv", sep="•", encoding='utf-8', index=False)


blob = TextBlob('harika')
if blob.sentiment.polarity > 0:
    print('positive')
elif blob.sentiment.polarity == 0:
    print('neutral')
else:
    print('negative')
print(blob)
