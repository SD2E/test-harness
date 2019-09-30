import os
import pandas as pd


from harness.utils.names import Names

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def get_result_csvs(loo_or_run_ids, th_output_location=None, file_type = 'testing'):
    '''
    Get result csv paths
    :param loo_or_run_ids: if a loo run, put the id of the loo run as key and the run_ids of that loo as a value
    if just a custom_run, make it a list
    :param th_output_location: Location of results
    :param file_type: must be either 'testing' or 'predicted'
    :return: get the prediction/testing csv paths
    '''
    print(file_type)
    assert file_type in ['testing','predicted'], 'file_type must be in ["testing", "predicted"]'
    for item in loo_or_run_ids:
        prediction_csv_paths = []
        if th_output_location is None:
            runs_path = os.path.join('test_harness_results', 'runs')
            previous_runs = []
            for this_run_folder in os.listdir(runs_path):
                if this_run_folder.rsplit("_")[1] in loo_or_run_ids:
                    print('{} was kicked off by this TestHarness instance. Its results will be submitted.'.format(this_run_folder))
                    if type(loo_or_run_ids)==list:
                        prediction_csv_path = os.path.join(runs_path, this_run_folder, file_type+'_data.csv')
                    else:
                        for run_id in loo_or_run_ids[item]:
                            prediction_csv_path = os.path.join(runs_path, this_run_folder,'loo_'+item, 'run_'+run_id,file_type + '_data.csv')
                    if os.path.exists(prediction_csv_path):
                        print("file found: ", prediction_csv_path)
                        prediction_csv_paths.append(prediction_csv_path)
                else:
                    previous_runs.append(this_run_folder)
            print('\nThe results for the following runs will not be submitted, '
                  'because they are older runs that were not initiated by this TestHarness instance:'
                  '\n{}\n'.format(previous_runs))
        else:
            if type(loo_or_run_ids) == list:
                prediction_csv_path = os.path.join(th_output_location,'test_harness_results','runs','run_'+run_id,file_type+'_data.csv')
                prediction_csv_paths.append(prediction_csv_path)
            else:
                for run_id in loo_or_run_ids[item]:
                    prediction_csv_path = os.path.join(th_output_location,'test_harness_results','runs','loo_'+item, 'run_' + run_id,
                                                       file_type + '_data.csv')
                    prediction_csv_paths.append(prediction_csv_path)
    return prediction_csv_paths

def query_leaderboard(th_output_location=None,LOO=False,query={}):
    '''
    Method that reads in all output prediction csvs from the leaderboard
    :param th_output_location: path to test harness output
    :param LOO = do you want to parse leave one out boards
    :param query_string: dictionary where keys are a column of leaderboard and values. Note that this is an AND across the conditions!
    :return: df_new: subset of leaderboard that matches query
    '''
    assert type(query) == dict, "query must of be of type dict. Column name must a column column in leaderboard."
    if th_output_location is None:
        leaderboard_path = os.path('test_harness_results')
    else:
        leaderboard_path = os.path.join(th_output_location,'test_harness_results')

    if LOO:
        leaderboard_df = pd.read_html(os.path.join(leaderboard_path,'loo_detailed_classification_leaderboard.html'))[0]
    else:
        leaderboard_df = pd.read_html(os.path.join(leaderboard_path,'custom_classification_leaderboard.html'))[0]
    for col in leaderboard_df.columns:
        if leaderboard_df[col].dtype == object:
            leaderboard_df[col]=leaderboard_df[col].str.replace("'","").str.replace("[","").str.replace("]","")
    if len(query)>0:
        temp_df = leaderboard_df.copy()
        for col in query:
            temp_df=temp_df[temp_df[col].str.contains(query[col])]
        return temp_df
    else:
        return leaderboard_df


if __name__ == '__main__':

    ##LOO Test
    th_location = '/Users/meslami/Documents/GitRepos/sd2-chaos/data/NovelChassis/NAND_20_Outputs/outputs_loo_part_condition'
    ##Example use of collecting test or predicted data
    paths = get_result_csvs(loo_or_run_ids={'52x6jbP3XX8AN':['dRb2j2jROwAQo','aPVG7Ray7mpQk','ml5YY9XNrrGwW']},th_output_location=th_location)
    print(paths)

    #Example of use of query
    query = {Names.MODEL_NAME:'random_forest',Names.TEST_GROUP:"timepoint: 5"}
    sub_df = query_leaderboard(th_location,True,query)
    print(sub_df.head(3))

    ##Custom Run Test
    th_location = None
    run_list = ['5dDzoa5ENJLyM','5Mr2mRqVbNAZN']
    paths = get_result_csvs(loo_or_run_ids=run_list)
    print(paths)

