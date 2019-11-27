import math
import pandas as pd
import copy

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores,test_ratings = subjects[0], subjects[1], subjects[2],subjects[3]
        neg_users, neg_items, neg_scores,neg_ratings = subjects[4], subjects[5], subjects[6], subjects[7]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'item': test_items,
                             'score': test_scores,
                            'ratings': test_ratings})
        # the full set
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores,
                            'ratings': neg_ratings+ test_ratings})
        print(full.shape)
        # get dict of golden items.
        self._test_items = { d['user'].iloc[0]:d['item'].to_list() for i,d in test.groupby('user')}
        # rank the items according to the scores for each user
        fullx = pd.concat(tuple([d.drop_duplicates(subset='item') for i,d in full.groupby('user')]),axis=0)
        
        full = copy.deepcopy(fullx)
        
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        score = 0.0
        # golden items hit in the top_K items
        score_1 = sum([(len(d[(d['item'].isin(self._test_items[d['user'].iloc[0]]))& (d['ratings']==1.0)])/self._top_k) for i,d in top_k.groupby('user')])
        score_2 = sum([(len(d[(d['item'].isin(self._test_items[d['user'].iloc[0]]))& (d['ratings']==0.0)])/self._top_k) for i,d in top_k.groupby('user')])
        score = score_1 - score_2
        return score/full['user'].nunique()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        score = 0.0
        score_1 = sum([sum(d[(d['item'].isin(self._test_items[d['user'].iloc[0]]))& (d['ratings']==1.0)]['rank'].apply(lambda x: math.log(2) / math.log(1 + x)).to_list()) for i,d in top_k.groupby('user')])
        score_2 = sum([sum(d[(d['item'].isin(self._test_items[d['user'].iloc[0]]))& (d['ratings']==0.0)]['rank'].apply(lambda x: math.log(2) / math.log(1 + x)).to_list()) for i,d in top_k.groupby('user')])
        score = score_1 - score_2
        return score / full['user'].nunique()
