README Last Updated by Hamed on 2/13/19


# Test Harness
###### Current Version: 3.2

## Why use the Test Harness?
1. Consistent results and outputs, both in terms of format and in terms of ensuring results are valid and without bugs
2. Collaboration and awareness of peer analytics through leaderboard summaries and access to model code
3. Automated calculation of various model performance metrics
4. Automated selection of future experiments to run (coming soon for protein design and perovskites)

## Using the Test Harness
You can use the Test Harness locally or with TACC resources such as Maverick2.

Python 3 should be used when using the Test Harness.

### Installation
1. Clone this repository into the environment of your choice (directory, conda env, virtualenv, etc)
2. Using command-line, navigate to the directory in which you cloned this repo (not inside the repo itself).
3. Run `pip3 install test-harness` or `pip3 install -e test-harness` .
This will install the `test-harness` package and make it visible to all other repositories/projects
you have in the current environment. The `-e` option stands for "editable". This will install the package
in a way where any local changes to the package will automatically be reflected in your environment.
See [this link](https://stackoverflow.com/questions/41535915/python-pip-install-from-local-dir/41536128)
for more details. 

**Note**: for some reason I cannot install the package on Maverick2 unless I use the `-e` option and also `--user`: `pip3 install -e test-harness --user`


### Running The Test Harness

First create a script file in your environment
(see the `example_scripts` folder for examples), then:
1. In your script, import modules from the `test-harness` package you installed earlier.
For example, `from harness.test_harness_class import TestHarness`
2. Create a TestHarness object with `output_location` path
3. Use the `run_custom` or `run_leave_one_out` methods
4. Results will be output in a `test_harness_results` folder whose location depends
on the path you give to `output_location` when initializing your TestHarness object.


#### Run Options
There are 2 types of runs currently supported in the Test Harness: Custom Runs and Leave-One-Out Runs
More details coming soon (probably in pydoc format)


#### Feature Extraction Options
Feature Extraction is controlled with the `feature_extraction` option. Valid values are:
1. False --> Feature Extraction is not carried out
2. True --> The `eli5_permutation` method is used
3. "eli5_permutation" --> Uses the [eli5 package's implementation of permutation importance for SkLearn Models](https://eli5.readthedocs.io/en/latest/autodocs/sklearn.html#eli5.sklearn.permutation_importance.PermutationImportance).
A more general version that isn't specific to sklearn models will be added in the future.
4. "rfpimp_permutation" --> Uses the [rfpimp package's implementation of permutation importance for Random Forest Models only](https://github.com/parrt/random-forest-importances).
5. "shap_audit" --> [SHapley Additive exPlanations](https://github.com/slundberg/shap)
6. "bba_audit" --> [Black Box Auditing by Haverford Team](https://github.com/algofairness/BlackBoxAuditing)

Feature Extraction results are saved in the file `feature_importances.csv` in the appropriate subfolder within the `results` folder.

**Note**: In a test that I ran, I got significantly different results from the two different implementations of permutation importance: this needs to be investigated.


### Maverick2 Instructions:

#### Setup:
1. `ssh _____@maverick2.tacc.utexas.edu`
2. Enter user/pass/auth-code
3. `module load tacc-singularity/2.6.0`
    1. Not necessary if you make singularity part of your default set of modules using "module save"
4. Only do these ONCE: first get on a compute node by using the `idev` command. Then run `singularity pull docker://eram5/test-harness:3.2`
    1. This is only needed the first time you're pulling the image, or if you want to pull an updated version of the image.
    2. This will build a singularity image from the docker container and store it in a ".simg" file under singularity_cache
    3. The path to the ".simg" file will be printed out, make sure to save the path for everytime you need to kick off a singularity shell!


#### Running interactively with `idev`:
1. `cd [wherever you're working]`. e.g. `cd $WORK` (`cdw` is equivalent) followed by `cd protein-design`
2. `idev -m 100` [or however many minutes you want the session to last]
    1. Note: if you want to keep a session going after you exit terminal (or get disconnected), you could use the `screen` command before `idev`
3. `singularity shell --nv /work/05260/hamed/singularity_cache/test-harness-3.2.simg` --> path is different for each user!
    1. The path is the one that was printed when you first pulled the docker container
    2. This starts an interactive singularity shell and places you within it

Note: if you try to install things within the singularity container,
you probably will have to add the "--user" parameter at the end,
e.g. `pip3 install pandas --user`. Ideally you would install all requirements within your dockerfile though

#### Running non-interactively using Sbatch/Slurm:
View the `sbatch_example.slurm` file in the `example_scripts` folder. As you can see the `sbatch_example.slurm` file is pretty self-explanatory.
In the last line you can see the command that runs `example_script.py`. In this case there is just one command being run, but you can add more if you would like.
Just keep in mind that the commands will be run in serial on a single node. To access multiple nodes and parallelization, you can either manually kick off more
jobs in the same way, or you can use launcher (see below for the guide to that).

1. The command to run a `slurm` file is `sbatch sbatch_example.slurm`. 
2. You can view the progress of your job in realtime by using `tail -f [name_of_my_job_id_file]`

#### Running non-interactively using Sbatch/Slurm, combined with parallelization using Launcher and Argparse:
Looking at the `example_scripts` folder, there are 3 files that are used for this:
1. `example_script_argparse.py` --> the script I am running
2. `jobfile_example` --> just a list of the jobs I want to run. Make sure you have an empty line at the end of this file or your last line won't be read!
3. `launcher_example.slurm` --> slurm file that uses Launcher to organize and split up tasks in the `jobfile` over multiple nodes so they can run in parallel

Once again you would use the `sbatch` and `tail` commands to kick off your jobs and view their progress.

** Please note that if you installed the `test-harness` package without the `-e` option,
then if you make any major changes to the `test-harness` package, you will have to
reinstall the package to update it.

### How It Works (Behind the Scenes)
The Test Harness consists of several parts that work together. 
1. Test Harness **Model** Classes and Instances: Basically just wrappers for your ML model so that the Test Harness can know how to run your model.
    1. All TH-Model Classes must subclass from the abstract base classes in `test_harness_models_abstract_classes.py`. A Model Class defines how to fit and predict.
    2. Once a TH-Model Class is created (e.g. SklearnClassification), TH-Model Instances can be created with specific models passed in.
2. `test_harness_class.py` contains the `TestHarness` class, which allows the user to add and execute different types of runs (custom, leave-one-out).
3. `run_classes.py` defines the `_BaseRun` class that is used by the `TestHarness` class to carry out runs.
4.  Script files that run the Test Harness. This is where you implement the Test Harness and use its methods to run your models.

As a user, you will mainly be concerned with the Test Harness `model classes/instances` (bullet 1 above), and the script files (bullet 4 above).


## Results and Leaderboards
By choosing an `output_location` when instantiating your `TestHarness` object,
you can dictate where the results of your runs go. For example if you want the results of
a certain set of runs to be isolated from other runs, you can set a previously unused
path to be your `output_location`.

For submitting to the SD2-wide leaderboard, set your `output_location` to the following path
in sd2e-community depending on your challenge problem:

1. Protein Design: `/sd2e-community/protein-design/test-harness-outputs`
2. Perovskites: tbd


## todo

To protect against breaking the test harness app: The best idea I have is to eventually have auto-deployer infra on Jenkins deploying the app on every merge to master.  It would try to deploy a dev app every time a merge request is made, and hopefully run tests on our dev path.  That way we’d catch errors before deploy to the prd version (edited) 
That’s not actually that hard to do once we have dev-escalation and I can set up a dev-test-harness-app
Ideally we’d set a rule on the repo that doesn’t allow merge to master without a merge request approver, and we can just put a merge request checklist in that asks if that test was successful.
I think that’s a decent balance between automation and process/rules

Implement merge request rules:
1. Version has been bumped
2. Changelog has been updated
3. Version tags on master
4. Someone has approved the merge request



