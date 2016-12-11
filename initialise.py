import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def initialise_with_lda(data, vocab_size, n_topics):
    """Initialise topic-word distribution and document-topic distribution

    Parameters
    ----------
    data : .csv file path
        Containing bag of words representations for documents
    n_samples : int
        Number of samples
    vocab_size : float
        Vocabulary size (V in the literature)
    n_topics : int
        Number of topics

    Returns
    -------
    (topic_word_matrix, doc_topic_matrix, vocabulary) :
        `topic_word_matrix` is the matrix of topic-word distributions
        `doc_topic_matrix` is the matrix of document-topic distributions
        `vocabulary` is an array of words
    """

    dataset = pd.read_csv(data, usecols=[1])
    data_samples = dataset['0'].tolist()

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=vocab_size,
                                    stop_words='english')

    tf = tf_vectorizer.fit_transform(data_samples)

    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    model = lda.fit(tf)

    vocabulary = tf_vectorizer.get_feature_names()
    doc_topic_matrix = lda.transform(tf)
    topic_word_matrix = model.components_

    return (topic_word_matrix, doc_topic_matrix, vocabulary)

def initialise_user_doc_matrix(data):
    """Initialise topic-word distribution and document-topic distribution

    Parameters
    ----------
    data : .csv file path
        as csv: Data frame with columns 'uid' (user ID), 'iid' (item ID) where a row represents
        a particular item being in a particular user's library.

    Returns
    -------
    user_ratings : 2-d array
        user_ratings[i,j] = 1 iff item j is in user i's library, 0 otherwise
    """

    dataset = pd.read_csv(data, header=0, sep=",")

    highest_item = max(dataset['iid'])
    highest_user = max(dataset['uid'])
    user_ratings = np.zeros((highest_item, highest_user))
    for i in range(0, len(dataset)-1):
        item = dataset.loc[i, 'iid']
        user = dataset.loc[i, 'uid']
        user_ratings[user, item] = 1
    return user_ratings
