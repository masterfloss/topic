pip install git+https://github.com/masterfloss/topic.git
from generatetopic import DataReader, TextDocumentCleaner, TopicModel, PESTELMapper, TopicEvolutionAnalyzer
site="https://github.com/masterfloss/bibliography/raw/refs/heads/main/”
f1  = site+”biblio_part_1.xlsx"
f2 = site+”biblio_part_2.xlsx"
f3 = site+”biblio_part_3.xlsx"
file_paths = [f1, f2, f3]
data_reader = DataReader(file_paths)
df = data_reader.read_data()
df['documents'] = df[['Title', 'Abstract', 'Author Keywords', 'Index Keywords']].fillna(").agg(‘'.join, axis=1)
cleaner = TextDocumentCleaner()
df = cleaner.transform(df)
topic_model = TopicModel(num_topics=6, max_features=3000)
topic_model.fit(df['cleaned_documents'])
topics = topic_model.get_topics()
PESTEL_KEYWORDS = {
'Political': ['policy', 'government', 'regulation', 'election', 'public policy', 'diplomacy', 'politics'],
'Economic': ['market', 'finance', 'trade', 'business', 'investment', 'economic growth', 'inflation', 'recession'],
'Social': ['society', 'culture', 'education', 'healthcare', 'social media', 'demographics', 'inequality'],
'Technological': ['technology', 'innovation', 'machine learning', 'artificial intelligence', 'automation', 'robotics'],
'Environmental': ['climate change', 'sustainability', 'pollution', 'recycling', 'carbon footprint', 'biodiversity'],
'Legal': ['law', 'regulation', 'compliance', 'intellectual property', 'human rights', 'litigation']
}
topic_mapper = PESTELMapper(PESTEL_KEYWORDS)
analyzer = TopicEvolutionAnalyzer(topic_model, topic_mapper)
topic_mapping = topic_mapper.transform(topics)
analyzer.analyze(df)
