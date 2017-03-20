'''

CTPF

'''

import logging

import math as math
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

class CTPF:
    def __init__(self, documents_path, n_topics=20, n_words=1000):
        '''
        CTPF

        Parameters
        ---------
        documents_path: .csv file path
            as csv: data frame containing bag of words representations of documents, indexed
            according to the item id's in the ratings
        n_topics : int
            Number of topics in LDA model
        '''

        self.PATH_TO_DOCUMENTS = documents_path

        self.n_topics = n_topics
        self.n_docs = 0
        self.n_words = n_words
        self.n_users = 0

        self.elbo = 0

        # Observed Variables
        self.documents = []
        self.ratings = 0
        self.test_ratings = 0
        self.predictions = 0
        self.words = 0

        # Latent Variables
        self.beta = 0
        self.theta = 0
        self.eta = 0
        self.epsilon = 0
        self.z = np.zeros((1, 1, 1))
        self.y = np.zeros((1, 1, 1))

        # Variational Parameters
        self.v_theta = {}
        self.v_beta = {}
        self.v_eta = {}
        self.v_epsilon = {}
        self.v_phi = np.zeros((1, 1, 1))
        self.v_ksi = np.zeros((1, 1, 1))

    def __initialise_with_lda(self, data, n_words, n_topics):
        """Initialise topic-word distribution and document-topic distribution

        Parameters
        ----------
        data : pandas data frame
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
        print 'Initialising with LDA'
        self.documents = pd.read_csv(data)
        data_samples = self.documents['0'].tolist()
        self.n_docs = len(data_samples)

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_words,
                                        stop_words='english')

        word_doc_matrix = tf_vectorizer.fit_transform(data_samples)

        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)

        model = lda.fit(word_doc_matrix)

        doc_topic_matrix = lda.transform(word_doc_matrix)
        topic_word_matrix = model.components_

        #normalise the rows
        doc_topic_matrix = normalize(doc_topic_matrix, axis=1, norm='l1')
        topic_word_matrix = normalize(topic_word_matrix, axis=1, norm='l1')

        return (topic_word_matrix, doc_topic_matrix, word_doc_matrix)

    def initialise_params(self):
        lda_init = self.__initialise_with_lda(self.PATH_TO_DOCUMENTS, self.n_words, self.n_topics)
        (self.beta, self.theta, self.words) = lda_init

        # TODO: initialise properly
        self.v_theta = {
            'shape': self.__initialise_eta_epsilon(self.n_docs, self.n_topics),
            'rate': self.__initialise_eta_epsilon(self.n_docs, self.n_topics)
        }

        # TODO: initialise properly
        self.v_beta = { 
            'shape': self.__initialise_eta_epsilon(self.n_words, self.n_topics),
            'rate': self.__initialise_eta_epsilon(self.n_words, self.n_topics)
        }

        self.v_eta = {
            'shape': self.__initialise_eta_epsilon(self.n_users, self.n_topics),
            'rate': self.__initialise_eta_epsilon(self.n_users, self.n_topics)
        }

        self.v_epsilon = {
            'shape': self.__initialise_eta_epsilon(self.n_docs, self.n_topics),
            'rate': self.__initialise_eta_epsilon(self.n_docs, self.n_topics)
        }

        self.v_phi = np.zeros((self.n_docs, self.n_words, self.n_topics))

        self.v_ksi = np.zeros((self.n_users, self.n_docs, 2 * self.n_topics))

    def __initialise_eta_epsilon(self, dim1, dim2):
        matrix = np.zeros((dim1, dim2))
        for element in np.nditer(matrix, op_flags=['readwrite']):
            element[...] = element + 0.3 + np.random.uniform(-0.1, 0.1)
        return matrix

    def initialise_user_doc_matrix(self, data):
        """Initialise user-document distribution

        Parameters
        ----------
        data : .csv file path
            as csv: Data frame with columns 'uid' (user ID), 'iid' (item ID), 'class' (train/test) where a row represents
            a particular item being in a particular user's library. 

        Returns
        -------
        user_ratings : numpy 2-d array
            user_ratings[i,j] = 1 iff item j is in user i's library, 0 otherwise
        """
        print 'Initialising user-document matrix...'
        dataset = pd.read_csv(data, header=0, sep=",")
        highest_item = max(dataset['iid'])
        highest_user = max(dataset['uid'])
        user_ratings = np.zeros((highest_user+1, highest_item+1))
        for i in range(0, len(dataset)-1):
            item = dataset.loc[i, 'iid']
            user = dataset.loc[i, 'uid']
            form = dataset.loc[i, 'class']
            if form == 'train':
                user_ratings[user, item] = 1

        self.ratings = user_ratings
        self.n_users = user_ratings.shape[0]

    def fit(self, ratings_path):
        '''Fit the model to the user ratings data.

        Parameters
        ----------
        ratings_path : .csv file path
            as csv: Data frame with columns 'uid' (user ID), 'iid' (item ID) where a row represents
            a particular item being in a particular user's library.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        self.initialise_user_doc_matrix(ratings_path)
        self.initialise_params()

        print 'Phi dims:', self.v_phi.shape, self.v_phi[0, 0]
        print 'Words dims:', self.words.shape, self.words[0, 0]
        print 'Ratings dims:', self.ratings.shape
        print 'Ksi dims:', self.v_ksi.shape
        print 'Eta dims:', self.v_eta['shape'].shape

        elbos = [1, 1]
        converged = False
        iteration = 0
        logging.info('Starting CAVI...')
        while not(converged): 
            iteration += 1
            logging.info(iteration)
            self.update_phi()
            #elbos.append(self.compute_elbo())
            self.update_ksi()
            #elbos.append(self.compute_elbo())
            self.__update_theta()
            #elbos.append(self.compute_elbo())
            self.__update_beta_shape()
            #elbos.append(self.compute_elbo())
            self.__update_beta_rate()
            #elbos.append(self.compute_elbo())
            self.__update_eta_shape()
            #elbos.append(self.compute_elbo())
            self.__update_eta_rate()
            #elbos.append(self.compute_elbo())
            self.__update_epsilon_shape()
            #elbos.append(self.compute_elbo())
            self.__update_epsilon_rate()
            #elbos.append(self.compute_elbo())
            if (iteration % 40 == 0):
                #elbos.append(self.compute_elbo())
                #if abs((elbos[-1] - elbos[-2]) / elbos[-2]) < 0.01:
                converged = True
                #logging.info(elbos)
                print 'Done! ELBO: ', elbos

        return self

    def __check_phi(self):
        # Checks that phi[d,v] sums to 1
        for d in range(self.v_phi.shape[0]):
            for v in range(self.v_phi.shape[1]):
                total = np.sum(self.v_phi[d, v, :])
                if (float(total) != 0):
                    print 'phi indices:', d, v,'| total:', total, '| vector:', self.v_phi[d, v, :]
    def __check_ksi(self):
        # Checks that ksi[u,d] sums to 1
        for u in range(self.v_ksi.shape[0]):
            for d in range(self.v_ksi.shape[1]):
                total = np.sum(self.v_ksi[u, d, :])
                if (int(total) != 0):
                    print 'ksi indices:', u, d,'| total:', total, '| vector:', self.v_ksi[u, d, :]

    def update_phi(self):
        print 'Updating phi...',
        for d in range(self.v_phi.shape[0]):
            for v in range(self.v_phi.shape[1]):
                if (int(self.words[d, v]) != 0):
                    update = np.zeros(self.v_phi.shape[2])
                    for k in range(self.v_phi.shape[2]):
                        print '(', d, ',', v, ',', k, ')\r',
                        update[k] = self.__multinomial_update(
                            self.v_theta['shape'][d, k],
                            self.v_theta['rate'][d, k],
                            self.v_beta['shape'][v, k],
                            self.v_beta['rate'][v, k]
                            )
                    update_normalised = update / update.sum()
                    self.v_phi[d, v] = update_normalised
        print 'Updating phi... done!'

    def update_ksi(self):
        print 'Updating ksi...',
        K = self.v_ksi.shape[2] / 2
        for u in range(self.v_ksi.shape[0]):
            for d in range(self.v_ksi.shape[1]):
                if (int(self.ratings[u, d]) != 0):
                    update = np.zeros(self.v_ksi.shape[2])
                    for k in range(self.v_ksi.shape[2]):
                        print '(', u, ',', d, ',', k, ')\r',
                        if k < K:
                            update[k] = self.__multinomial_update(
                                self.v_eta['shape'][u, k],
                                self.v_eta['rate'][u, k],
                                self.v_theta['shape'][d, k],
                                self.v_theta['rate'][d, k]
                                )
                        else:
                            update[k] = self.__multinomial_update(
                                self.v_eta['shape'][u, k - K],
                                self.v_eta['rate'][u, k - K],
                                self.v_epsilon['shape'][d, k - K],
                                self.v_epsilon['rate'][d, k - K]
                                )
                    update_normalised = update / update.sum()
                    self.v_ksi[u, d] = update_normalised
        print '\rUpdating ksi... done!'

    def __update_theta(self):
        print 'Updating theta (shape)...',
        for d in range(self.v_theta['shape'].shape[0]):
            for k in range(self.v_theta['shape'].shape[1]):
                self.__update_theta_shape(d, k)
                self.__update_theta_rate(d, k)
        print 'Done!'

    def __update_theta_shape(self, d, k):
        r_ksi = self.ratings[:, d].dot(self.v_ksi[:, d, k])
        w_phi = self.words.toarray()[d, :].dot(self.v_phi[d, :, k])
        self.v_theta['shape'][d, k] = 0.3 + r_ksi + w_phi

    def __update_theta_rate(self, d, k):
        self.v_theta['rate'][d, k] = \
                    0.3 \
                    + np.dot(self.v_beta['shape'][:, k], 1. / self.v_beta['rate'][:, k]) \
                    + np.dot(self.v_eta['shape'][:, k], 1. / self.v_eta['rate'][:, k])

    def __update_beta_shape(self):
        print 'Updating beta (shape)...',
        for v in range(self.v_beta['shape'].shape[0]):
            for k in range(self.v_beta['shape'].shape[1]):
                w_phi = np.dot(self.words.toarray()[:, v], self.v_phi[:, v, k])
                self.v_beta['shape'][v, k] = 0.3 + w_phi
        print 'Done!'

    def __update_beta_rate(self):
        print 'Updating beta (rate)...',
        for v in range(self.v_beta['rate'].shape[0]):
            for k in range(self.v_beta['rate'].shape[1]):
                self.v_beta['rate'][v, k] = \
                    0.3 \
                    + self.v_theta['shape'][:, k].dot(1. / self.v_theta['rate'][:, k])
        print 'Done!'

    def __update_eta_shape(self):
        print 'Updating eta (shape)...',
        _K = self.v_ksi.shape[2] / 2
        for i in range(self.v_eta['shape'].shape[0]):
            for j in range(self.v_eta['shape'].shape[1]):
                self.v_eta['shape'][i, j] = \
                    0.3 \
                    + np.dot(self.ratings[i, :], self.v_ksi[i, :, j]) \
                    + np.dot(self.ratings[i, :], self.v_ksi[i, :, _K + j])
        print 'Done!'

    def __update_eta_rate(self):
        print 'Updating eta (rate)...',
        for i in range(self.v_eta['rate'].shape[0]):
            for j in range(self.v_eta['rate'].shape[1]):
                self.v_eta['rate'][i, j] = \
                    0.3 \
                    + np.dot(self.v_theta['shape'][:, j], 1. / self.v_theta['rate'][:, j]) \
                    + np.dot(self.v_epsilon['shape'][:, j], 1. / self.v_epsilon['rate'][:, j])
        print 'Done!'

    def __update_epsilon_shape(self):
        print 'Updating epsilon (shape)...',
        _K = self.v_ksi.shape[2] / 2
        for i in range(self.v_epsilon['shape'].shape[0]):
            for j in range(self.v_epsilon['shape'].shape[1]):
                self.v_epsilon['shape'][i, j] = \
                    0.3 \
                    + np.dot(self.ratings[:, i], self.v_ksi[:, i, _K + j])
        print 'Done!'

    def __update_epsilon_rate(self):
        print 'Updating epsilon (rate)...',
        for i in range(self.v_epsilon['rate'].shape[0]):
            for j in range(self.v_epsilon['rate'].shape[1]):
                self.v_epsilon['rate'][i, j] = \
                    0.3 \
                    + np.dot(self.v_eta['shape'][:, j], 1. / self.v_eta['rate'][:, j])
        print 'Done!'

    def __multinomial_update(self, theta_shp, theta_rte, beta_shp, beta_rte):
        return np.exp(
            sp.special.digamma(theta_shp)
            - math.log(theta_rte)
            + sp.special.digamma(beta_shp)
            - math.log(beta_rte)
        )
    def test_elbo(self):
        for i in range(5):
            beta_elbo = self.__elbo_gamma_sum(0.3, 0.3, self.v_beta)
            print '\n\nTesting beta updates. Beta ELBO contribution:', beta_elbo
            self.__update_phi()
            self.__update_ksi()
            self.__update_theta_shape()
            self.__update_theta_rate()
            self.__update_eta_shape()
            self.__update_eta_rate()
            self.__update_epsilon_shape()
            self.__update_epsilon_rate()
            self.__update_beta_shape()
            self.__update_beta_rate()
    def compute_elbo(self):
        elbo = \
            self.__elbo_gamma_sum(0.3, 0.3, self.v_beta) \
            + self.__elbo_gamma_sum(0.3, 0.3, self.v_theta) \
            + self.__elbo_gamma_sum(0.3, 0.3, self.v_eta) \
            + self.__elbo_gamma_sum(0.3, 0.3, self.v_epsilon) \
            + self.__elbo_poisson_w_sum(self.v_theta, self.v_beta) \
            + self.__elbo_poisson_r_sum(self.v_theta, self.v_eta, self.v_epsilon) \
            + self.__elbo_multi_y_sum(self.v_eta, self.v_theta, self.v_epsilon, self.v_ksi) \
            + self.__elbo_multi_z_sum(self.v_phi, self.v_beta, self.v_theta, self.ratings) \
            - self.__elbo_v_gamma_sum(self.v_beta) \
            - self.__elbo_v_gamma_sum(self.v_theta) \
            - self.__elbo_v_gamma_sum(self.v_eta) \
            - self.__elbo_v_gamma_sum(self.v_epsilon) \
            - self.__elbo_v_multi_r_sum(self.ratings, self.v_ksi) \
            - self.__elbo_v_multi_w_sum(self.words, self.v_phi)
        self.elbo = elbo
        return elbo
    def __elbo_gamma_sum(self, a, b, variable):
        sum = 0
        for i in range(variable['shape'].shape[0]):
            for j in range(variable['shape'].shape[1]):
                sum += self.__elbo_gamma_term(a, b, variable['shape'][i, j], variable['rate'][i, j])
        logging.info('Gamma sum contribution: %s', sum)
        return sum
    def __elbo_gamma_term(self, a, b, shape, rate):
        return (a - 1) * (sp.special.digamma(shape) - math.log(rate)) - b * shape / rate
    def __elbo_poisson_w_sum(self, theta, beta):
        sum = 0
        for d in range(self.v_phi.shape[0]):
            for v in range(self.v_phi.shape[1]):
                for k in range(self.v_phi.shape[2]):
                    sum -= (theta['shape'][d, k] / theta['rate'][d, k]) \
                            * (beta['shape'][v, k] / beta['rate'][v, k])
        logging.info('Poisson sum contribution: %s', sum)
        return sum
    def __elbo_poisson_r_sum(self, theta, eta, epsilon):
        sum = 0
        for u in range(self.v_ksi.shape[0]):
            for d in range(self.v_ksi.shape[1]):
                for k in range(self.v_ksi.shape[2] / 2):
                    sum -= (eta['shape'][u, k] / eta['rate'][u, k]) \
                            * (theta['shape'][d, k] / theta['rate'][d, k] \
                            + epsilon['shape'][d, k] / epsilon['rate'][d, k])
        return sum
    def __elbo_multi_y_sum(self, eta, theta, epsilon, ksi):
        sum = 0
        for u in range(self.v_ksi.shape[0]):
            for d in range(self.v_ksi.shape[1]):
                for k in range(self.v_ksi.shape[2]):
                    sum += self.__elbo_multi_y_term(u, d, k, eta, theta, epsilon, ksi)
        logging.info('Multi sum contribution: %s', sum)
        return sum
    def __elbo_multi_y_term(self, u, d, i, eta, theta, epsilon, ksi):
        K = ksi.shape[2] / 2
        k = i % K
        r = self.ratings[u, d]
        ksi = self.v_ksi[u, d, i]
        gamma1 = self.__exp_log_gamma(eta['shape'][u, k], eta['rate'][u, k])
        if (k < K):
            gamma2 = self.__exp_log_gamma(theta['shape'][d, k], theta['rate'][d, k])
        else:
            gamma2 = self.__exp_log_gamma(epsilon['shape'][d, k], epsilon['rate'][d, k])

        return r * ksi * (gamma1 + gamma2)

    def __elbo_multi_z_sum(self, phi, beta, theta, ratings):
        sum = 0
        for d in range(self.v_phi.shape[0]):
            for v in range(self.v_phi.shape[1]):
                for k in range(self.v_phi.shape[2]):
                    sum += self.__elbo_multi_z_term(self.words[d, v], self.v_phi[d, v, k],
                    self.v_theta['shape'][d, k], self.v_theta['rate'][d, k],
                    self.v_beta['shape'][v, k], self.v_beta['rate'][v, k])
        logging.info('Multi sum contribution: %s', sum)
        return sum
    def __elbo_multi_z_term(self, w, phi, theta_shape, theta_rate, beta_shape, beta_rate):
        return w * phi * (self.__exp_log_gamma(theta_shape, theta_rate) + self.__exp_log_gamma(beta_shape, beta_rate))
    def __exp_log_gamma(self, shape, rate):
        return sp.special.digamma(shape) - math.log(rate)
    def __elbo_v_gamma_sum(self, variable):
        sum = 0
        for i in range(variable['shape'].shape[0]):
            for j in range(variable['shape'].shape[1]):
                sum += self.__elbo_v_gamma_term(variable['shape'][i, j], variable['rate'][i, j])
        logging.info('vGamma sum contribution: %s', sum)
        return sum
    def __elbo_v_gamma_term(self, shape, rate):
        if float(shape) == 0.: return 0
        if float(rate) == 0.: return 0
        result = shape * math.log(rate) - math.log(sp.special.gamma(shape)) + (shape - 1)*(sp.special.digamma(shape) - math.log(rate)) - shape
        if np.isfinite(result):
            return result
        else:
            return 0

    def __elbo_v_multi_w_sum(self, words, phi):
        sum = 0
        for d in range(phi.shape[0]):
            for v in range(phi.shape[1]):
                for k in range(phi.shape[2]):
                    sum += self.__elbo_v_multi_w_term(words[d, v], phi[d, v, k])
        logging.info('vMulti sum contribution: %s', sum)
        return sum
    def __elbo_v_multi_w_term(self, w, phi):
        if phi == 0: return 0
        return w * phi * np.log(phi)
    def __elbo_v_multi_r_sum(self, ratings, ksi):
        sum = 0
        for u in range(ksi.shape[0]):
            for d in range(ksi.shape[1]):
                for k in range(ksi.shape[2] / 2):
                    sum += self.__elbo_v_multi_r_term(ratings[u, d], ksi[u, d, k])
        logging.info('vMulti sum contribution: %s', sum)
        return sum
    def __elbo_v_multi_r_term(self, r, ksi):
        if ksi == 0: return 0
        return r * ksi * np.log(ksi)
    def test(self, data):
        print 'Initialising test ratings matrix...'
        dataset = pd.read_csv(data, header=0, sep=",")
        highest_item = max(dataset['iid'])
        highest_user = max(dataset['uid'])
        user_ratings = np.zeros((highest_user+1, highest_item+1))
        for i in range(0, len(dataset)-1):
            item = dataset.loc[i, 'iid']
            user = dataset.loc[i, 'uid']
            form = dataset.loc[i, 'class']
            if form == 'test':
                user_ratings[user, item] = 1
        self.test_ratings = user_ratings
        self.predictions = self.__predict_ratings()
        self.report = self.__generate_report(self.test_ratings, self.predictions)
        print 'Average Precision (10):', np.mean(self.report['precision']['10']), ' | ', 'Average Recall:', np.mean(self.report['recall']['10'])
    def __predict_ratings(self):
        predictions = np.zeros((self.v_ksi.shape[0], self.v_ksi.shape[1]))
        for u in range(self.v_ksi.shape[0]):
            for d in range(self.v_ksi.shape[1]):
                item_rating = np.sum(self.v_ksi[u, d, :])
                predictions[u, d] = item_rating
        return predictions
    def __generate_report(self, test_ratings, predictions):
        precision = {}
        recall = {}
        for num_items in range(10):
            n = int((num_items + 1) * 2)
            precisions = []
            recalls = []
            for u in range(test_ratings.shape[0]):
                sorted_predictions = np.argsort(predictions[u, :])
                indices = sorted_predictions[:n]

                predicted_subset = np.zeros(len(sorted_predictions))
                predicted_subset[indices] = 1
                
                user_precision = test_ratings[u, :].dot(predicted_subset) / n
                user_recall = test_ratings[u, :].dot(predicted_subset) / np.sum(test_ratings[u, :])
                if math.isnan(user_recall): user_recall = 0

                precisions.append(user_precision)
                recalls.append(user_recall)

            precision[str(n)] = np.mean(precisions)
            recall[str(n)] = np.mean(recalls)
        return {'precision': precision, 'recall': recall}
            

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="sunday_night_log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    print "creating CTPF instance"
    CTPF = CTPF('data/1k/documents.csv', 15, 750)
    print "attempting to fit"
    CTPF.fit('data/1k/ratings.csv')
    print "attempting to test"
    CTPF.test('data/1k/ratings.csv')
    logging.info(CTPF.report)
