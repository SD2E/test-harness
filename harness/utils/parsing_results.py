import os
import pandas as pd
import sklearn.metrics as metrics



from harness.utils.names import Names

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def get_leaderboard(th_output_location=None, loo=True, classification=True):
    '''
    Get the leaderboard as a df
    :param th_output_location: test harness output locato
    :param loo: True/False -- is this a LOO Run
    :param classification: True/False -- is this a Classification or Regression task
    :return: the leaderboard in the form of a dataframe
    '''
    if th_output_location is None:
        th_output_location = os.getcwd()
    th_output_location = os.path.join(th_output_location,'test_harness_results')

    if loo:
        if classification:
            leaderboard = pd.read_html(os.path.join(th_output_location,Names.LOO_FULL_CLASS_LBOARD+'.html'))[0]
        else:
            leaderboard = pd.read_html(os.path.join(th_output_location,Names.LOO_FULL_REG_LBOARD+'.html'))[0]
    else:
        if classification:
            leaderboard = pd.read_html(os.path.join(th_output_location,Names.CUSTOM_CLASS_LBOARD+'.html'))[0]
        else:
            leaderboard = pd.read_html(os.path.join(th_output_location,Names.CUSTOM_REG_LBOARD+'.html'))[0]
    return leaderboard


def get_result_csv_paths(loo_or_run_ids, th_output_location=None, file_type=Names.TESTING_DATA):
    '''
    Get result csv paths
    :param loo_or_run_ids: if a loo run, put the id of the loo run as key and the run_ids of that loo as a value list
    if just a custom_run, make it a list
    :param th_output_location: Location of you gave to the test harness
    :param file_type: must be in Names.OUTPUT_FILE keys
    :return: get the output csv paths
    '''
    assert file_type in Names.OUTPUT_FILES, 'file_type must be in {0}'.format(Names.OUTPUT_FILES.keys())
    for item in loo_or_run_ids:
        output_csv_paths = []
        if th_output_location is None:
            runs_path = os.path.join('test_harness_results', 'runs')
            previous_runs = []
            for this_run_folder in os.listdir(runs_path):
                if this_run_folder.rsplit("_")[1] in loo_or_run_ids:
                    print('{} was kicked off by this TestHarness instance. Its results will be collected.'.format(this_run_folder))
                    if type(loo_or_run_ids)==list:
                        output_csv_path = os.path.join(runs_path, this_run_folder, Names.OUTPUT_FILES[file_type])
                    else:
                        for run_id in loo_or_run_ids[item]:
                            output_csv_path = os.path.join(runs_path, this_run_folder,'loo_'+item, 'run_'+run_id,Names.OUTPUT_FILES[file_type])
                    if os.path.exists(output_csv_path):
                        print("file found: ", output_csv_path)
                        output_csv_paths.append(output_csv_path)
                else:
                    previous_runs.append(this_run_folder)

        else:
            if type(loo_or_run_ids) == list:
                output_csv_path = os.path.join(th_output_location,'test_harness_results','runs','run_'+item,Names.OUTPUT_FILES[file_type])
                output_csv_paths.append(output_csv_path)
            else:
                for run_id in loo_or_run_ids[item]:
                    output_csv_path = os.path.join(th_output_location,'test_harness_results','runs','loo_'+item, 'run_' + run_id,
                                                       Names.OUTPUT_FILES[file_type])
                    output_csv_paths.append(output_csv_path)
    return output_csv_paths



def get_incorrect_classification_results_query(query, th_output_location, loo=False, file_type=Names.TESTING_DATA):
    '''
    This method queries the leaderboard, retrieves the files of the associated run_ids, subsets that
    :param query: query to subset the leaderboard
    :param th_output_location: path to output location
    :param loo: Boolean. If a leave one out runn.
    :param classification: Boolean. True if classification
    :param file_type: one of the output of the files
    :return:
    '''
    df_leaderboard_sub = query_leaderboard(query=query,th_output_location=th_output_location,loo=loo,classification=True)
    col_to_predict = df_leaderboard_sub[Names.COLUMN_PREDICTED].unique()
    if len(col_to_predict)>1:
        raise RuntimeError('This function can only be used when you have a single predicted column. You currently have {0}'.format(len(col_to_predict)))
    print("Column to predict",col_to_predict)

    paths = get_result_csv_paths_query(query=query,th_output_location=th_output_location,loo=loo,file_type=file_type,classification=True)
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.loc[~(df[col_to_predict[0]] == df[col_to_predict[0]+'_predictions'])]
        df['path']=path
        dfs.append(df)
    df_all = pd.concat(dfs)
    return df_all

def get_correct_classification_results_query(query, th_output_location, loo=False, file_type=Names.TESTING_DATA):
    '''
    This method queries the leaderboard, retrieves the files of the associated run_ids, subsets that
    :param query: query to subset the leaderboard
    :param th_output_location: path to output location
    :param loo: Boolean. If a leave one out runn.
    :param classification: Boolean. True if classification
    :param file_type: one of the output of the files
    :return:
    '''
    df_leaderboard_sub = query_leaderboard(query=query,th_output_location=th_output_location,loo=loo,classification=True)
    col_to_predict = df_leaderboard_sub[Names.COLUMN_PREDICTED].unique()
    if len(col_to_predict)>1:
        raise RuntimeError('This function can only be used when you have a single predicted column. You currently have {0}'.format(len(col_to_predict)))
    print("Column to predict",col_to_predict)

    paths = get_result_csv_paths_query(query=query,th_output_location=th_output_location,loo=loo,file_type=file_type,classification=True)
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.loc[(df[col_to_predict[0]] == df[col_to_predict[0]+'_predictions'])]
        df['path']=path
        dfs.append(df)
    df_all = pd.concat(dfs)
    return df_all

def get_roc_curve_query(query, th_output_location, loo=False, file_type=Names.TESTING_DATA):
    '''
    This method queries the leaderboard, retrieves the files of the associated run_ids, subsets that
    :param query: query to subset the leaderboard
    :param th_output_location: path to output location
    :param loo: Boolean. If a leave one out run.
    :param classification: Boolean. True if classification
    :param file_type: one of the output of the files
    :return:
    '''
    df_leaderboard_sub = query_leaderboard(query=query,th_output_location=th_output_location,loo=loo,classification=True)
    col_to_predict = df_leaderboard_sub[Names.COLUMN_PREDICTED].unique()
    if len(col_to_predict)>1:
        raise RuntimeError('This function can only be used when you have a single predicted column. You currently have {0}'.format(len(col_to_predict)))
    print("Column to predict",col_to_predict)
    paths = get_result_csv_paths_query(query=query,th_output_location=th_output_location,loo=loo,file_type=file_type,classification=True)
    dfs = []
    for path in paths:
        df_preds = pd.read_csv(path)
        y_true = df_preds[col_to_predict[0]].tolist()
        y_probas = df_preds[col_to_predict[0]+'_prob_predictions'].tolist()
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_probas)
        ###TODO: NEED TO FIGURE OUT WHAT TO DO WITH THIS!

def get_incorrect_classification_results(loo_or_run_ids, th_output_location, loo=False, file_type=Names.TESTING_DATA):
    '''
    This method queries the leaderboard, retrieves the files of the associated run_ids, subsets that
    :param loo_or_run_ids: dictionary of loo/run ids
    :param th_output_location: path to output location
    :param loo: Boolean. If a leave one out runn.
    :param classification: Boolean. True if classification
    :param file_type: one of the output of the files
    :return: the
    '''
    df_leaderboard_list = []
    for item in loo_or_run_ids:
        query={}
        if loo:
            query[Names.LOO_ID]=item
            query[Names.RUN_ID]=loo_or_run_ids[item]
        else:
            query[Names.RUN_ID]=item
        df_sub = get_incorrect_classification_results_query(query=query,th_output_location=th_output_location,loo=loo, \
                                                            file_type=file_type)
        df_leaderboard_list.append(df_sub)
    df_all = pd.concat(df_leaderboard_list)
    return df_all


def get_result_csv_paths_query(query, th_output_location, loo=False, classification=False,file_type=Names.TESTING_DATA):
    '''
    Get the results of a leaderboard from a query
    :param query: query
    :param th_output_location: test harness location
    :param loo: Boolean of whether you are querying a loo run
    :param classification: Boolean of whether this was a classification test
    :param file_type: the file type you want to return
    :return: list of paths associated with the query
    '''
    df_sub = query_leaderboard(query, th_output_location, loo=loo, classification=classification)
    if loo:
        loo_or_run_ids={}
        for ind in df_sub.index:
            loo_id = df_sub.loc[ind,Names.LOO_ID]
            run_id = df_sub.loc[ind,Names.RUN_ID]
            if loo_id not in loo_or_run_ids:
                loo_or_run_ids[loo_id]=[]
            loo_or_run_ids[loo_id].append(run_id)
    else:
        loo_or_run_ids = df_sub[Names.RUN_ID].tolist()
    paths = get_result_csv_paths(loo_or_run_ids=loo_or_run_ids,th_output_location=th_output_location,file_type=file_type)
    return paths



def query_leaderboard(query, th_output_location, loo=False, classification=False):
    '''
    Method that reads in all output prediction csvs from the leaderboard
    :param query: dictionary where keys are a column of leaderboard and values. Note that this is an AND across the conditions!

    :param th_output_location: path to test harness output
    :param loo: True/False -- is this a LOO Run
    :param classification: True/False -- is this a Classification or Regression task
    :return: df_new: subset of leaderboard that matches query
    '''
    assert type(query) == dict, "query must of be of type dict. Column name must a column column in leaderboard."
    leaderboard_df = get_leaderboard(th_output_location, loo, classification)
    for col in leaderboard_df.columns:
        if leaderboard_df[col].dtype == object:
            leaderboard_df[col] = leaderboard_df[col].str.replace("'", "").str.replace("[", "").str.replace("]", "")
    if len(query) > 0:
        temp_df = leaderboard_df.copy()
        for col in query:
            temp_df = temp_df[temp_df[col].str.contains(query[col])]
        return temp_df
    else:
        return leaderboard_df


if __name__ == '__main__':
    th_location = '/Users/meslami/Documents/GitRepos/test-harness/example_scripts/Data_Sharing_Demo'

    # Example of use of query to subset leaderboard
    query = {Names.MODEL_NAME:'random_forest',Names.TEST_GROUP:"topology: EHEE"}
    sub_df = query_leaderboard(query, th_location, loo=True,classification=True)

    # Example of use of query to get incorrect results
    query = {Names.MODEL_NAME: 'random_forest', Names.TEST_GROUP: "topology: EHEE"}
    sub_df = get_incorrect_classification_results_query(query, th_location, loo=True)
    print(sub_df.head)


