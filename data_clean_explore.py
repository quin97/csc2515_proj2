import json
import pandas as pd
import os
import gc
import numpy as np
from collections import Counter, defaultdict
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import re
import regex
# import altair as alt
import csv
from sklearn import model_selection
import itertools

os.chdir('/Users/quinxie/Downloads/csc2515/csc2515-rating-prediction/project2')

df2=pd.read_csv("data.csv", sep = "\t", index_col=0)
print(df2.shape)
df2.head()
df2.columns
# df2 = df2[~pd.isnull(df2['abstract'])]
# df4 = df2[:70]
# # ============================================metadata exploration================================================
#
# # clean up author column
df2["author_clean"] = df2["author"].apply(lambda x: regex.sub(r'\([^()]*+(?:(?R)[^()]*)*+\)', '', x)) # recursively remove nested parentheses and its content
df2["author_clean"] = df2["author_clean"].apply(lambda x: re.sub("\sand\s",",",x)) # sub and with comma
df2["author_clean"] = df2["author_clean"].apply(lambda x: re.sub(", et al","",x)) # sub and with comma
df2["author_clean"] = df2["author_clean"].apply(lambda x: re.sub('(,){2,}',",",x))
df2["author_clean"] = df2["author_clean"].apply(lambda x: x.split(','))
def get_clean(x):
    # Remove all characters not in the English alphabet
    x = re.sub("[^\w\s]", "", str(x))#  to remove all special characters other than words and white spaces
    x = str(x).lower()
    x = re.sub(r'\s+', ' ',x).strip()
    return x

df2["author_clean"] = df2["author_clean"].apply(lambda x: [get_clean(a) for a in x if get_clean(a).strip()])
#
# # def flatten(s):
# #     if s == []:
# #         return s
# #     if isinstance(s[0], list):
# #         return flatten(s[0]) + flatten(s[1:])
# #     return s[:1] + flatten(s[1:])
# #
# # df2["author"] = df2["author"].apply(lambda x: flatten(x))
# df2["author"] = df2["author"].apply(lambda x: list(itertools.chain.from_iterable(x)))

df2["author_clean"] = df2["author_clean"].apply(lambda x: (",").join(x))
df2.to_csv("data_clean_author.csv", sep = "\t")
df2=pd.read_csv("data_clean_author.csv", index_col=0)
df2["author"] = df2["author_clean"].apply(lambda x: x.split(','))


allauthors = np.concatenate(df2["author"])
allauthors_clean = set(allauthors)
print("Total number of authors: "+str(len(allauthors_clean)))
## Don't run!
# num_paper_per_author = [[x,allauthors.count(x)] for x in allauthors_clean]
# with open("num_paper_per_author.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(num_paper_per_author)

# only get rows with doi info
df3 = df2[~pd.isnull(df2['doi'])]
print(df3.shape)
# check no. of papers with journal-ref
numjournals = df2[~pd.isnull(df2['journal'])].shape
# check no.of paper with both info
journal_and_doi = df3[~pd.isnull(df3['journal'])].shape
alljournals = df3["doi"].apply(lambda x: re.match("(.*?)/",x).group())
alljournalset = list(set(alljournals))
print("Total number of journals: "+str(len(alljournalset)))

df2['general_category'] = df2.categories.apply(lambda x: x.split('.')[0])
allcats = list(set(df2['general_category']))
print("Total number of categories: "+str(len(allcats)))

print(df2["categories"].describe())
print(df2["general_category"].describe())
# https://arxiv.org/help/api/user-manual
category_map = {'astro-ph': 'Astrophysics',
'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
'astro-ph.EP': 'Earth and Planetary Astrophysics',
'astro-ph.GA': 'Astrophysics of Galaxies',
'astro-ph.HE': 'High Energy Astrophysical Phenomena',
'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
'astro-ph.SR': 'Solar and Stellar Astrophysics',
'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
'cond-mat.mtrl-sci': 'Materials Science',
'cond-mat.other': 'Other Condensed Matter',
'cond-mat.quant-gas': 'Quantum Gases',
'cond-mat.soft': 'Soft Condensed Matter',
'cond-mat.stat-mech': 'Statistical Mechanics',
'cond-mat.str-el': 'Strongly Correlated Electrons',
'cond-mat.supr-con': 'Superconductivity',
'cs.AI': 'Artificial Intelligence',
'cs.AR': 'Hardware Architecture',
'cs.CC': 'Computational Complexity',
'cs.CE': 'Computational Engineering, Finance, and Science',
'cs.CG': 'Computational Geometry',
'cs.CL': 'Computation and Language',
'cs.CR': 'Cryptography and Security',
'cs.CV': 'Computer Vision and Pattern Recognition',
'cs.CY': 'Computers and Society',
'cs.DB': 'Databases',
'cs.DC': 'Distributed, Parallel, and Cluster Computing',
'cs.DL': 'Digital Libraries',
'cs.DM': 'Discrete Mathematics',
'cs.DS': 'Data Structures and Algorithms',
'cs.ET': 'Emerging Technologies',
'cs.FL': 'Formal Languages and Automata Theory',
'cs.GL': 'General Literature',
'cs.GR': 'Graphics',
'cs.GT': 'Computer Science and Game Theory',
'cs.HC': 'Human-Computer Interaction',
'cs.IR': 'Information Retrieval',
'cs.IT': 'Information Theory',
'cs.LG': 'Machine Learning',
'cs.LO': 'Logic in Computer Science',
'cs.MA': 'Multiagent Systems',
'cs.MM': 'Multimedia',
'cs.MS': 'Mathematical Software',
'cs.NA': 'Numerical Analysis',
'cs.NE': 'Neural and Evolutionary Computing',
'cs.NI': 'Networking and Internet Architecture',
'cs.OH': 'Other Computer Science',
'cs.OS': 'Operating Systems',
'cs.PF': 'Performance',
'cs.PL': 'Programming Languages',
'cs.RO': 'Robotics',
'cs.SC': 'Symbolic Computation',
'cs.SD': 'Sound',
'cs.SE': 'Software Engineering',
'cs.SI': 'Social and Information Networks',
'cs.SY': 'Systems and Control',
'econ.EM': 'Econometrics',
'eess.AS': 'Audio and Speech Processing',
'eess.IV': 'Image and Video Processing',
'eess.SP': 'Signal Processing',
'gr-qc': 'General Relativity and Quantum Cosmology',
'hep-ex': 'High Energy Physics - Experiment',
'hep-lat': 'High Energy Physics - Lattice',
'hep-ph': 'High Energy Physics - Phenomenology',
'hep-th': 'High Energy Physics - Theory',
'math.AC': 'Commutative Algebra',
'math.AG': 'Algebraic Geometry',
'math.AP': 'Analysis of PDEs',
'math.AT': 'Algebraic Topology',
'math.CA': 'Classical Analysis and ODEs',
'math.CO': 'Combinatorics',
'math.CT': 'Category Theory',
'math.CV': 'Complex Variables',
'math.DG': 'Differential Geometry',
'math.DS': 'Dynamical Systems',
'math.FA': 'Functional Analysis',
'math.GM': 'General Mathematics',
'math.GN': 'General Topology',
'math.GR': 'Group Theory',
'math.GT': 'Geometric Topology',
'math.HO': 'History and Overview',
'math.IT': 'Information Theory',
'math.KT': 'K-Theory and Homology',
'math.LO': 'Logic',
'math.MG': 'Metric Geometry',
'math.MP': 'Mathematical Physics',
'math.NA': 'Numerical Analysis',
'math.NT': 'Number Theory',
'math.OA': 'Operator Algebras',
'math.OC': 'Optimization and Control',
'math.PR': 'Probability',
'math.QA': 'Quantum Algebra',
'math.RA': 'Rings and Algebras',
'math.RT': 'Representation Theory',
'math.SG': 'Symplectic Geometry',
'math.SP': 'Spectral Theory',
'math.ST': 'Statistics Theory',
'math-ph': 'Mathematical Physics',
'nlin.AO': 'Adaptation and Self-Organizing Systems',
'nlin.CD': 'Chaotic Dynamics',
'nlin.CG': 'Cellular Automata and Lattice Gases',
'nlin.PS': 'Pattern Formation and Solitons',
'nlin.SI': 'Exactly Solvable and Integrable Systems',
'nucl-ex': 'Nuclear Experiment',
'nucl-th': 'Nuclear Theory',
'physics.acc-ph': 'Accelerator Physics',
'physics.ao-ph': 'Atmospheric and Oceanic Physics',
'physics.app-ph': 'Applied Physics',
'physics.atm-clus': 'Atomic and Molecular Clusters',
'physics.atom-ph': 'Atomic Physics',
'physics.bio-ph': 'Biological Physics',
'physics.chem-ph': 'Chemical Physics',
'physics.class-ph': 'Classical Physics',
'physics.comp-ph': 'Computational Physics',
'physics.data-an': 'Data Analysis, Statistics and Probability',
'physics.ed-ph': 'Physics Education',
'physics.flu-dyn': 'Fluid Dynamics',
'physics.gen-ph': 'General Physics',
'physics.geo-ph': 'Geophysics',
'physics.hist-ph': 'History and Philosophy of Physics',
'physics.ins-det': 'Instrumentation and Detectors',
'physics.med-ph': 'Medical Physics',
'physics.optics': 'Optics',
'physics.plasm-ph': 'Plasma Physics',
'physics.pop-ph': 'Popular Physics',
'physics.soc-ph': 'Physics and Society',
'physics.space-ph': 'Space Physics',
'q-bio.BM': 'Biomolecules',
'q-bio.CB': 'Cell Behavior',
'q-bio.GN': 'Genomics',
'q-bio.MN': 'Molecular Networks',
'q-bio.NC': 'Neurons and Cognition',
'q-bio.OT': 'Other Quantitative Biology',
'q-bio.PE': 'Populations and Evolution',
'q-bio.QM': 'Quantitative Methods',
'q-bio.SC': 'Subcellular Processes',
'q-bio.TO': 'Tissues and Organs',
'q-fin.CP': 'Computational Finance',
'q-fin.EC': 'Economics',
'q-fin.GN': 'General Finance',
'q-fin.MF': 'Mathematical Finance',
'q-fin.PM': 'Portfolio Management',
'q-fin.PR': 'Pricing of Securities',
'q-fin.RM': 'Risk Management',
'q-fin.ST': 'Statistical Finance',
'q-fin.TR': 'Trading and Market Microstructure',
'quant-ph': 'Quantum Physics',
'stat.AP': 'Applications',
'stat.CO': 'Computation',
'stat.ME': 'Methodology',
'stat.ML': 'Machine Learning',
'stat.OT': 'Other Statistics',
'stat.TH': 'Statistics Theory'}
categories_names = set(category_map.keys())
set(df2["categories"]).difference(categories_names) ## empty set
categories_names.difference(set(df2["categories"]))

a = df2.groupby(["categories"]).count()
print(a["id"].describe())
cm = plt.cm.get_cmap('RdYlBu_r')
Y,X = np.histogram((a["id"]), 149, normed=True)
x_span = X.std()
C = [cm(((x-X.min())/x_span)) for x in X]
# plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
# # log transform X and Y
# plt.yscale('log')
# plt.xscale('log')
# plt.title("Histogram of paper categories");
# plt.xlabel("Category");
# plt.ylabel("Number of papers");
a = df2["general_category"].value_counts().plot(kind='bar',figsize=(14,8),title="Histogram of paper categories", color=C)
a.set_xlabel("Category")
a.set_ylabel("Number of papers")
plt.show()

# alt chart can't show in python console
# paper_per_cat= df2[["id","general_category"]].groupby(["general_category"]).sum()
# bar = alt.Chart(paper_per_cat).mark_bar().encode(
#     x='general_category',
#     y=alt.Y('id'),
#     tooltip = ['general_category','id']
# ).properties(width = 600
#              ).configure_axis(grid=False
#                               ).configure_view(strokeOpacity=0).interactive()

# below the commented out code adapted from
# https://www.kaggle.com/artgor/arxiv-metadata-exploration
# authors_per_paper = defaultdict()
# df2[["id","author"]].set_index("id").to_dict("list", into = authors_per_paper)

# year_pattern = r'([1-2][0-9]{3})'
# # year_abstract_words = {}
# year_categories = {}
# year_authors = {}
# for i in range(df2.shape[0]):
#     paper = df2.iloc[i]
#     if paper['update_date']:
#         year = re.match(year_pattern, paper['update_date']).groups() if re.match(year_pattern, paper['update_date']) \
#             else None
#         if year:
#             year = [int(i) for i in year if int(i) < 2020 and int(i) >= 1991]
#             if year == []:
#                 year = None
#             else:
#                 year = min(year)
#     else:
#         year = None
#     if year:
#         if year not in year_categories.keys():
#             # year_abstract_words[year] = defaultdict(int)
#             year_categories[year] = defaultdict(int)
#             year_authors[year] = defaultdict(int)
#         # collect counts of various things over years
#     # for word in paper['abstract'].replace('\n', ' ').split():
#     #     if year:
#     #         year_abstract_words[year][word] += 1
#     if paper['general_category']:
#         if year:
#             year_categories[year][paper['general_category']] += 1
#     paper_authors = [x for x in paper["author"] if x == x]
#     if paper_authors:
#         if year:
#             for author in paper_authors:
#                 year_authors[year][author] += 1
#
# # year_abstract_words_df = pd.DataFrame(year_abstract_words)
# # year_abstract_words_df.to_csv("year_abstract_words.csv")
#
# year_authors_df = pd.DataFrame(year_authors)
# year_authors_df.to_csv("year_authors.csv")
# # # to index a column:
# # year_authors_df.iloc[:,0]
#
#
# year_categories_df = pd.DataFrame(year_categories)
# year_categories_df.to_csv("year_categories.csv")
#
# ### run lines below on google colab
# year_authors_df = pd.read_csv("year_authors.csv",index_col=0)
# authors = []
# # names_to_exclude = ['dubna', 'japan','infn','department of physics',
# #                     'the netherlands','trieste','bonn','ukraine','heidelberg',
# #                     'australia','stsci','eso','korea','switzerland', 'israel',
# #                     'canada','mexico', 'caltech','poland', 'cambridge', 'spain',
# #                     'garching', 'india', 'uk', 'et al', 'moscow', 'usa', 'france',
# #                     'russia', 'italy', 'germany','brazil','astronomy',
# #                     'd  collaboration', 'the opal collaboration','the babar collaboration',
# #                     'madrid', 'berkeley', 'mit', 'astrophysics', 'cfa', 'bangalore',
# #                     'chile', 'princeton', 'china', 'nasa gsfc', 'beijing', 'argentina',
# #                     'edinburgh','nrao','roma']
# for col in year_authors_df.columns:
#     top_authors = [i for i in year_authors_df[col].fillna(0).sort_values().index][-10:]
#     # top_authors = [x for x in top_authors if x == x] # dont need it anymore b/c name.strip() in get_clean
#     authors.extend(top_authors)
# authors = list(set(authors))

# year_authors_df1 = year_authors_df.T[authors]
# year_authors_df1 = year_authors_df1.sort_index()
# year_authors_df2 = year_authors_df1.reset_index().melt(id_vars=['index'])
# year_authors_df2.columns = ['year', 'author', 'count']
# fig = px.line(year_authors_df2, x="year", y="count", color='author', width=1500, height=1000)
# fig.show()
# img_bytes = fig.to_image(format="png")
#

# # year_authors_df = pd.read_csv("year_authors.csv",index_col=0)
# cats = []
# for col in year_categories_df.columns:
#     top_cats = [i for i in year_categories_df[col].fillna(0).sort_values().index]
#     cats.extend(top_cats)
# cats = list(set(cats))
#
# year_categories_df1 = year_categories_df.T[cats]
# year_categories_df1 = year_categories_df1.sort_index()
# year_categories_df2 = year_categories_df1.reset_index().melt(id_vars=['index'])
# year_categories_df2.columns = ['year', 'category', 'count']
# fig = px.line(year_categories_df2, x="year", y="count", width = 1500, height = 900, color='category')
# fig.show()


## takes long time to run
## cat_authors = dict.fromkeys(allcats,[]) # this is wrong
# cat_authors = {x: set() for x in allcats}
# for i in range(df2.shape[0]):
#     paper = df2.iloc[i]
#     # print ("paper", paper)
#     paper_authors = set(paper["author"])
#     cat = paper['general_category']
#     cat_authors[cat]=cat_authors[cat].union(paper_authors)
#
# author_cat = pd.DataFrame(list(cat_authors.items()),columns = ['categories','authors'])
# author_cat['authors'] = author_cat['authors'].apply(lambda x: list(x))
# author_cat['authors'] = author_cat['authors'].apply(lambda x: ",".join(x))
# author_cat.to_csv("authors_by_cat.csv",sep = "\t")
author_cat=pd.read_csv("authors_by_cat.csv", sep= "\t", index_col=0)
author_cat["authors"] = author_cat['authors'].apply(lambda x: x.split(' , '))

author_cat["num_authors"] = author_cat['authors'].apply(lambda x: len(x))
print(author_cat["num_authors"].describe().round(2))

## plot with seaborn instead
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# toplot = author_cat[["categories","num_authors"]]
# toplot.set_index("categories", inplace = True)
# cm = plt.cm.get_cmap("RdYlBu_r")
# cNorm  = colors.Normalize(vmin=0, vmax=toplot["num_authors"].max())
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
# a = toplot.plot(kind='bar',figsize=(14,8),title="Histogram of authors per categories", color=scalarMap.to_rgba(toplot["num_authors"]))
# a.set_xlabel("Category")
# a.set_ylabel("Number of authors")
# plt.show()

import seaborn as sns
tips = sns.load_dataset("tips")
sns.set_theme(font_scale=5)

tips
fig = sns.catplot(data=author_cat[["categories","num_authors"]], x="categories",y = "num_authors", kind = "bar",height=20, aspect=1.618,palette="ch:.25" )
fig.set_xticklabels(rotation=90)
fig.set(ylabel = "number of authors",title = "Histogram of number of authors per category")


def countWords(review):
    return len(review.replace('\n', ' ').split())
df3 = df2
df3["countWordsAbs"] = df3["abstract"].astype(str).apply(countWords)
print(df3["countWordsAbs"].describe().round(2))
df4 = df3.loc[df3['countWordsAbs'] >= 10]

# df_train, df_test = model_selection.train_test_split(df4, test_size=0.2,random_state = 10)
# df_train.to_csv("train.csv")
# df_test.to_csv("test.csv")

bins=range(0,350,50)
a = list(np.clip(df3["countWordsAbs"], 0, 400))
fig, ax = plt.subplots(1,1)
ax.hist(a, bins=bins, align='left', color="royalblue", rwidth=0.7)
plt.title("Histogram of number of words per abstract");
plt.xlabel("Number of words");
plt.ylabel("Number of papers");
plt.show()
