from faulthandler import cancel_dump_traceback_later
import pyodbc
import pandas as pd
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

ft = fasttext.load_model('Models/FastText/cc.fa.300.bin')

def get_token_name(token_Id):
    return pyodbc.connect(con_str_Posts).cursor().execute("SELECT Token FROM Tokens Where Id = " + str(token_Id)).fetchone()[0]

if __name__ == '__main__':

    # Connection String Posts 
    con_str_Posts = 'DRIVER={SQL Server};SERVER=.;DATABASE=Posts;UID=sa;PWD=abc@123'

    stopwords = []

    cursor = pyodbc.connect(con_str_Posts).cursor()
    cursor = cursor.execute(
    """SELECT Id
    FROM Tokens
    WHERE Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 14)
    AND Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 15)
    AND Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 16)
    AND Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 17)
    AND Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 18)
    AND Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 37)
    AND Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 38)
    AND Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 39)
    AND Id IN (SELECT Token_Id FROM [Post-Token] JOIN Posts ON Post_Seq = Seq WHERE WindowNum = 40)""")

    for row in cursor:
        stopwords.append(row[0])



    # Connection String Posts 
    con_str_Posts = 'DRIVER={SQL Server};SERVER=.;DATABASE=Posts;UID=sa;PWD=abc@123'

    windows = dict()

    # windows_Num = [14,15,16,17,18,37,38,39,40]

    windows_Num = [16]

    for window_num in windows_Num:

        patterns = []

        Tokens = []

        # window_num = 16

        RunId = pyodbc.connect(con_str_Posts).cursor().execute("SELECT Id FROM Runs Where WinNum = " + str(window_num)).fetchone()[0]

        cursor = pyodbc.connect(con_str_Posts).cursor()
        cursor = cursor.execute("""SELECT DISTINCT [Token_Id]
                                    FROM [Post-Token] JOIN Posts ON  [Post_Seq] = [Seq]
                                    WHERE WindowNum = """ + str(window_num))

        for row in cursor:
            if row[0] not in stopwords:
                Tokens.append(row[0])

        for Token in Tokens:
            candidate_pattern = [Token]

            pattern = []

            while candidate_pattern != []:
                Tok =  candidate_pattern.pop()

                if Tok not in pattern:
                    pattern.append(Tok)
                
                    cursor = pyodbc.connect(con_str_Posts).cursor()
                    cursor = cursor.execute('EXEC Get_Max_AGF ?, ?', Tok, RunId)

                    for row in cursor:
                        if row[0] not in candidate_pattern and row[0] not in pattern:
                            candidate_pattern.append(row[0])

            if len(pattern) > 1:
                patterns.append(pattern)
        
        patterns.sort(key=lambda x: len(x), reverse=True)

        i = 0
        while i < len(patterns):
            j = i+1
            while j < len(patterns):
                if all(item in patterns[i] for item in patterns[j]):
                    patterns.remove(patterns[j])
                j += 1
            i += 1
        
        patterns_name = []

        for pattern in patterns:
            patterns_name.append([get_token_name(p) for p in pattern])

        # df = pd.DataFrame(zip(range(len(patterns_name)), patterns_name), columns=['Id', 'Pattern']) 
        # df.to_excel('patterns_name_win' + str(window_num) + '.xlsx', index= False)

        patterns_emb = []

        for pattern in patterns:
            emb = []
            for p in pattern:
                emb.append(ft.get_word_vector(get_token_name(p)))

            if emb != []:
                patterns_emb.append(np.average(emb,axis=0))

        sim = [] 

        for i in range(0, len(patterns_emb)):
            for j in range(0, len(patterns_emb)):
                pattern_i = patterns_emb[i].reshape(1, -1)
                pattern_j = patterns_emb[j].reshape(1, -1)
                sim.append([i, j, cosine_similarity(pattern_i, pattern_j)[0][0]])

        # df = pd.DataFrame(sim, columns=['i', 'j', 'sim'])
        # df.to_excel('patterns_sim_win' + str(window_num) + '.xlsx', index= False)

        FP_list = range(len(patterns))

        def Dis_func(fst, Scnd):
            for s in sim:
                if s[0] == fst and s[1] == Scnd:
                    return 1 - s[2]
            return 1
        
        clusterer = hdbscan.HDBSCAN(metric = Dis_func)
        labels = clusterer.fit_predict(np.array(FP_list).reshape(-1, 1))

        C = dict()

        for l,label in enumerate(labels):
            if label not in C:
                C.update({label:[]})
            C[label].append(FP_list[l])

        # df = pd.DataFrame(zip(C.keys(),C.values()))
        # df.to_excel('patterns_cluster_win' + str(window_num) + '.xlsx', index= False)

        data = []

        for c in C: 
            for i in C[c]:
                data.append([c, i, [get_token_name(p) for p in patterns[i]]])

        df = pd.DataFrame(data)#, columns=['Tok1', 'Tok2', 'AGF'])
        df.to_excel('patterns__win' + str(window_num) + '.xlsx', index= False)

        meh = 0