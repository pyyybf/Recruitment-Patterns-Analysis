import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

html_stop_words = ['nbsp', 'quot', 'amp', 'lt', 'gt', 'apos',
                   'middot', 'ldquo', 'rdquo', 'lsquo', 'rsquo', 'sbquo', 'ndash', 'mdash', 'hellip', 'bull', 'pr', 'laquo', 'raquo']
